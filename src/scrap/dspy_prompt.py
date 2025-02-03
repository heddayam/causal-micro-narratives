import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune, SignatureOptimizer, COPRO, MIPRO
from dspy.teleprompt.vanilla import LabeledFewShot
# from rouge import Rouge 
from datasets import load_metric
from src.utils import utils
import os
import argparse
import json
from pathlib import Path
# from dspy.modules import anthropic

# rouge = load_metric('rouge')

# def score_rouge(gold, pred, trace=None):
#     gold = gold['syllabus']
#     pred = pred['syllabus']
#     rouge_score = scorer.score(gold, pred)
#     # rouge_score = rouge.compute(predictions=[gold], references=[pred], rouge_types=['rouge2'], use_stemmer=True, use_aggregator=True) #['rouge1', 'rouge2', 'rougeL']
#     # r1 = rouge_score['rouge1'].mid
#     r2 = rouge_score['rouge2'].fmeasure
#     # rl = rouge_score['rougeL'].mid
#     # breakpoint()
#     return r2

def true_false_check(gold, pred, trace=None):
    # breakpoint()
    if gold['contains_narrative'] == "False":
        return "False" in pred['contains_narrative']
    
    if gold['contains_narrative'] == "True":
        return "True" in pred['contains_narrative']
    
    narr_preds = pred.narratives.split(",")
    narr_gold = gold.narratives.split(",")

    overlap = set(narr_preds).intersection(set(narr_gold))
    if len(overlap) > 0:
        return 1

    # for item in gold.narratives:
    #     if item['cause'] is not None and item['cause'] in pred.narratives:
    #         return 1
    #     if item['effect'] is not None and item['effect'] in pred.narratives:
    #         return 1
    return 0

    # gold = gold['cited']
    # pred = True if "yes" in pred['cited'].lower() else False
    # return gold == pred

def run_dspy(dev_ds, test_ds, model, fewshot=False):
    if model == 'claude':
        os.environ['ANTHROPIC_API_KEY'] = 'API_KEY'
        model = dspy.dsp.anthropic.Claude(model='claude-3-sonnet-20240229', max_tokens=200)#, model_type='chat')
        # model = dspy.dsp.anthropic.Claude(model='claude-3-opus-20240229', max_tokens=200)#, model_type='chat')
        NUM_THREADS = 1
    elif model == 'mistral':
        model = dspy.HFModel(model = 'mistralai/Mistral-7B-Instruct-v0.1', max_tokens=200)
        NUM_THREADS = 32
    else:
        os.environ['OPENAI_API_KEY'] = "API_KEY"
        # turbo = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=200, model_type='chat')
        model = dspy.OpenAI(model='gpt-3.5-turbo-16k', max_tokens=200, model_type='chat')
        NUM_THREADS = 32

    dspy.settings.configure(lm=model)
    # Toggling this to true will redo the bootstrapping process. When
    # it is set to False, the existing demonstrations will be used but
    # turbo will still be used to evaluate the zero-shot and full programs.
    RUN_FROM_SCRATCH = False

    # train = []
    # train_syllabuses = []
    # for instance in syllabus_examples:
    #     train.append((instance['opinion'], instance['syllabus']))
    #     train_syllabuses.append(instance['syllabus'])

    # train = [dspy.Example(opinion=opinion, syllabus=syllabus).with_inputs('sentence') for opinion, syllabus in train]

    dev = []
    for instance in dev_ds:
        dev.append((instance['text'], instance['foreign'], instance['contains-narrative'], instance['inflation-narratives']))
    
    # dev = [dspy.Example(sentence=text, foreign=foreign, contains_narrative=contains, narratives=narratives).with_inputs('sentence') for text, foreign, contains, narratives in dev]
    dev = [dspy.Example(sentence=text, contains_narrative=contains).with_inputs('sentence') for text, foreign, contains, narratives in dev]


    test = []
    for instance in test_ds:
        test.append((instance['text'], instance['foreign'], instance['contains-narrative'], instance['inflation-narratives']))

    # test = [dspy.Example(sentence=text, foreign=foreign, contains_narrative=contains, narratives=narratives).with_inputs('sentence') for text, foreign, contains, narratives in test]
    test = [dspy.Example(sentence=text, contains_narrative=contains).with_inputs('sentence') for text, foreign, contains, narratives in test]


    # Define a dspy.Predict module with the signature `question -> answer` (i.e., takes a question and outputs an answer).
    # predict = dspy.Predict('sentence -> foreign,contains_narrative,narratives')

    # Use the module!
    # x = predict(src='mourad is great', target='mourad is not great')
    # x = predict(sentence="If we had not controlled inflation, our families would have been spending 35-40 per cent more, the Finance Minister said.")
    # breakpoint()



    class RelevanceSignature(dspy.Signature):
        # f"""{Path("../prompt-templates/in-context/claude/instruction_simple.txt").read_text()}"""
        """Review the provided news sentence. Determine if the sentence implies a cause/effect of inflation. Reply 'True' if it does, and reply 'False' if it doesn't suggest either a cause or effect of inflation."""

        sentence = dspy.InputField(desc="news sentence for analysis", prefix="News Sentence:")
        # foreign = dspy.OutputField(desc="Boolean indicating if the sentence is about a non-US country or economy.", prefix="Foreign:")
        contains_narrative = dspy.OutputField(desc="Boolean indicating whether the sentence contains an inflation narrative or not.", prefix="Contains Narrative:")
        # narratives = dspy.OutputField(desc="The labels of inflation narratives in the sentence, if any.", prefix="Narratives:")


    # print(x)
    class NoCoT(dspy.Module):  # let's define a new module
        def __init__(self, optimized):
            super().__init__()

            # here we declare the chain of thought sub-module, so we can later compile it (e.g., teach it a prompt)
            # self.generate_answer = dspy.ChainOfThought(SummarizerSignature)
            if optimized:
                self.generate_answer = dspy.Predict(RelevanceSignatureOptimized)
            else:
                # breakpoint()
                self.generate_answer = dspy.Predict(RelevanceSignature)
        
        def forward(self, sentence):
            return self.generate_answer(sentence=sentence)  # here we use the module
        
    class CoT(dspy.Module):
        def __init__(self, optimized):
            super().__init__()

            if optimized:
                self.generate_answer = dspy.ChainOfThought(RelevanceSignatureOptimized)
            else:
                # breakpoint()
                self.generate_answer = dspy.ChainOfThought(RelevanceSignature)
        
        def forward(self, sentence):
            return self.generate_answer(sentence=sentence)  # here we use the module
        
    # metric_EM = dspy.evaluate.answer_exact_match
        
    # NUM_THREADS = 32
    evaluate = False
    if evaluate:
        evaluate = Evaluate(devset=dev, metric=true_false_check, num_threads=NUM_THREADS, display_progress=True, display_table=0)


        eval_out = evaluate(NoCoT(optimized=False))
        breakpoint()

   

    kwargs = dict(num_threads=NUM_THREADS, display_progress=True, display_table=1) # Used in Evaluate class in the optimization process

    if fewshot:
        # teleprompter = BootstrapFewShotWithRandomSearch(
        #     metric=true_false_check, 
        #     max_bootstrapped_demos=0, 
        #     max_labeled_demos=20,
        #     # verbose=True
        # )
        teleprompter = MIPRO(
            metric=true_false_check,
            verbose=True
            # breadth=
        )
        # labeled_fewshot_optimizer = LabeledFewShot(k=20)
        # compiled_prompt_opt = labeled_fewshot_optimizer.compile(NoCoT(optimized=False), trainset=dev)
        compiled_prompt_opt = teleprompter.compile(NoCoT(optimized=False), trainset=dev, eval_kwargs=kwargs, max_labeled_demos=20, max_bootstrapped_demos=5, num_trials=3)
    else:
        teleprompter = COPRO(
            metric=true_false_check,
            verbose=True
            # breadth=
        )
        compiled_prompt_opt = teleprompter.compile(NoCoT(optimized=False), trainset=dev, eval_kwargs=kwargs)
       

    breakpoint()
    
    class RelevanceSignatureOptimized(dspy.Signature):
        """Review the provided Supreme Court opinion text. Deliver a concise, neutral summary that captures the essence of the legal reasoning, main points of law, conclusions drawn, and the implications of the decision, all whilst adhering to comprehensible language suitable for an educated general audience."""
        # context = dspy.InputField()
        opinion = dspy.InputField(desc="SCOTUS Opinion", prefix="Opinion:")
        syllabus = dspy.OutputField(desc="SCOTUS Syllabus", prefix="Summary of Supreme Court Opinion:")

    # NUM_THREADS = 32
    evaluate = Evaluate(devset=test, metric=true_false_check, num_threads=NUM_THREADS, display_progress=True, display_table=0)


    eval_summs = evaluate(NoCoT(optimized=True))

    res = eval_summs(cot_compiled)
    
    for t in test:
        compiled_prompt_opt(t['opinion'])

    breakpoint()

    # teleprompter = BootstrapFewShot(metric=metric_EM, max_bootstrapped_demos=2)
    # cot_compiled = teleprompter.compile(CoT(), trainset=train)

    # x = cot_compiled("If we had not controlled inflation, our families would have been spending 35-40 per cent more, the Finance Minister said.")

    # print(x)
    # breakpoint()
    # #turbo.inspect_history(n=1)

    # NUM_THREADS = 32
    # evaluate_narratives = Evaluate(devset=dev, metric=metric_EM, num_threads=NUM_THREADS, display_progress=True, display_table=15)

    # res = evaluate_narratives(cot_compiled)
    # print
    # breakpoint()

def get_output_fields(instance):
    out = utils.reconstruct_training_input(instance)
    out = json.loads(out)
    # breakpoint()
    out['foreign'] = str(out['foreign'])
    out['contains-narrative'] = str(out['contains-narrative'])

    classes = []
    if out['inflation-narratives'] is not None:
        out['inflation-narratives'] = out['inflation-narratives']['narratives']
        for narr in out['inflation-narratives']:
            # breakpoint()
            if 'cause' in narr:
                classes.append(narr['cause'])
            if 'effect' in narr:
                classes.append(narr['effect'])
        out['inflation-narratives'] = ",".join(classes)
    return out


def main(model):
    # ds = utils.load_hf_dataset(type='relevance', balanced=True, small=True)
    ds = utils.load_hf_dataset()

    # syllabus_examples = ds['train'].shuffle(seed=42).select(list(range(10)))
    train_ds = ds['train'].shuffle(seed=42).select(list(range(100)))
    test_ds = ds['test'].shuffle(seed=42)#.select(list(range(10)))

    train_ds = train_ds.map(get_output_fields)
    test_ds = test_ds.map(get_output_fields)

    run_dspy(train_ds, test_ds, model)

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--model', choices=['claude', 'mistral', 'gpt'], default='claude')
    args = parser.parse_args()
    main(args.model)