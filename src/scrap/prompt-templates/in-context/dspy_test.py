import dspy
from dspy.evaluate import Evaluate
from dspy.teleprompt import BootstrapFewShot, BootstrapFewShotWithRandomSearch, BootstrapFinetune

turbo = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=250, model_type='chat')

dspy.settings.configure(lm=turbo)

gpt4T = dspy.OpenAI(model='gpt-4-1106-preview', max_tokens=350, model_type='chat')

# Toggling this to true will redo the bootstrapping process. When
# it is set to False, the existing demonstrations will be used but
# turbo will still be used to evaluate the zero-shot and full programs.
RUN_FROM_SCRATCH = False

train = [
    ("But, for one thousand today you cannot buy the one thousand tomorrow's goods because of inflation.", "purchase"),
    ("Needle of a disposable syringe with a drop of liquid on the tip against the background of a fragment...+ of a US dollar bill getty To take the last point first, the original wartime profits tax was part of a larger effort to control inflation in an economy almost wholly given over to war production.", "fiscal"),
    ("For example, if the 10-year Treasury currently has an interest rate of x% while a similar maturity TIPS has a rate of y% we might imply that expected inflation is x--y%.", "none"),
    ("This would be wise, and not just because a tightening labor market will gradually put upward pressure on wages and inflation.", "supply"),
    ("Tasos Katopodis/Pool via AP) WASHINGTON--For decades, the Federal Reserve made clear its readiness to raise interest rates at the earliest signs of creeping inflation.", "rates"),
    ("Experts have also been warning of massive inflation since the Federal Reserve began quantitative easing.", "monetary"),
    ("For those in 30 occupational categories (chosen because 'we thought they would be interesting and understandable '), Planet Money plotted their current income against their median family income in 1979, adjusted for inflation.", "none"),
    ("The increased stipend will undoubtedly be a welcome handout, but amid accelerating inflation rates and rising gas prices, it may still not be enough", "cost"),
    ("As of this writing, the nominal FAO Food Price Index (uncorrected for inflation) has remained above 200 for more than three years, the likely product of rising demand, speculation, and ethanol production.", "demand,supply"),
    ("If the central bank actions aimed at addressing financial stability risks are large and persistent, the inflation rate will likely deviate from target for many years.", "monetary"),
    ("Nancy Davis, who manages the Quadratic Interest Rate Volatility and Inflation Hedge ETF, said she has seen a jump in interest in recent weeks from investors worried about the impact of rebound in some of the consumer and asset prices quashed by this year's crisis.", "savings")
]

train = [dspy.Example(sentence=sentence, answer=answer).with_inputs('sentence') for sentence, answer in train]

dev = [
    ("(Editing by Elaine Hardcastle) Next In Business News WASHINGTON U.S. consumer prices rose more than expected in August as healthcare costs recorded their biggest gain in 32-1/2 years, pointing to a steady build-up of inflation that could allow the Federal Reserve to raise interest rates this year.", "cost,rates"),
    ("A key policy intent of the tax was to offset the damaging effects of the longstanding federal tax preference for employer-sponsored insurance (ESI), one of which is to drive excess health cost inflation.", "fiscal"),
    ("UK inflation to move even higher: Pro James Knightley, U.K. economist at ING Wholesale Banking, reacts to the latest U.K. inflation data and says it is likely to climb a bit higher as the slack in the economy shrinks.", "demand"),
    ("Accounting for inflation, that $321 would be worth about $900 in today's dollars.", "purchase"),
    ("In the late 1960s, as inflation and other factors increased the cost of employer-sponsored health benefits, employers began instituting annual deductibles and coinsurance on their health benefits plans, and/or excluding coverage for certain medical items that were legally allowed to be covered by IRS regulations (e.g. vision, dental, alternative medicine).", "cost-push"),
    ("TOKYO, Nov 15 The dollar traded within sight of its highest level in more than 13-1/2 years on Tuesday as bond yields soared on expectations that President-elect Donald Trump's economic policies will fuel inflation.", "fiscal,savings"),
    ("The Bank of Japan's unprecedented stimulus and Abe's pro-growth reforms have yet to spur a recovery in inflation and gross domestic product growth, and the country is yet again in recession.", "monetary,fiscal"),
    ("Inflation More than 40 years ago, respected economist Paul Volcker took over as Chair of the Fed and raised interest rates to control inflation.", "rates"),
    ("Of course, it's impossible to predict with certainty what will happen with inflation or interest rates--or, frankly, anything.", "none"),
    ("He also mentioned the plight of the auto industry and the increase of retail inflation.", "none"),
    ("Second, when you make goods and services, ANY goods and services, a false 'right,' meaning that you have the right to the labor and property of others, you necessarily and inevitably have skyrocketing inflation in the price of those goods and services.", "expect")
]

dev = [dspy.Example(sentence=sentence, answer=answer).with_inputs('sentence') for sentence, answer in dev]

# Define a dspy.Predict module with the signature `question -> answer` (i.e., takes a question and outputs an answer).
predict = dspy.Predict('sentence -> answer')

# Use the module!
x = predict(sentence="If we had not controlled inflation, our families would have been spending 35-40 per cent more, the Finance Minister said.")

class EconNarrativeSignature(dspy.Signature):
    ("""Below are lists of causes and effects of inflation, delimted by XML tags. Read every single one, including the subcategories within them. """
    """You will be provided a sentence (delimited with XLM tags) for which you will be tasked to identify causes/effects of inflation. Focus on what is directly said in the sentence. In other words, focus on direct causes and/or effects of inflation and identify the categories that best describe them. """
    """\n<causes>
[demand] Demand-side factors: Pull-side or demand-pull inflation.
[supply] Supply-side factors: Push-side or cost-push inflation.
[wage] Built-in wage inflation: Also known as wage inflation or wage-price spiral. 
[monetary] Monetary factors: Central bank policies that contribute to inflation.
[fiscal] Fiscal factors: Government policies that contribute to inflation.
[expect] Expectations: The expectation that inflation will rise often leads to a rise in inflation.
[international] International Trade and Exchange Rates: International trade and exchange rate factors that can cause inflation.
[other-cause] Other Causes: Causes not included in above.
</causes>
<effects>
[purchase] Reduced Purchasing Power: Inflation erodes the purchasing power of money (such as the U.S. dollar) over time.
[cost] Cost of Living Increases: Inflation can raise the cost of living, particularly impacting individuals on fixed incomes, pensioners, and those with lower wages.
[uncertain] Uncertainty Increases: Inflation can create uncertainty about future prices (or future inflation itself), particularly if the inflation is high or unpredictable.
[rates] Interest Rates Raised: Central banks may respond to inflation by raising interest rates to curb spending and investment.
[redistribution] Income or Wealth Redistribution: Inflation can redistribute income and wealth between people in the economy.
[savings] Impact on Savings: Inflation can affect various types of savings/financial investments.
[trade] Impact on Global Trade: Inflation can impact a country's trade or competitiveness in global markets.
[cost-push] Cost-Push on Businesses: Rising costs of production due to inflationary pressures can squeeze business profits, potentially leading to reduced investment, job cuts and unemployment, or higher prices for consumers.
[social] Social and Political Impact: Inflation can have social and political economic implications.
[govt] Government Policy and Public Finances Impact: Inflation may impact government spending policies or programs.
[other-effect] Other Effects: Effects not included in above.
</effects>"""
    )

    # context = dspy.InputField()
    sentence = dspy.InputField()
    answer = dspy.OutputField(desc="cause/effect")

# print(x)
class CoT(dspy.Module):  # let's define a new module
    def __init__(self):
        super().__init__()

        # here we declare the chain of thought sub-module, so we can later compile it (e.g., teach it a prompt)
        self.generate_answer = dspy.ChainOfThought(EconNarrativeSignature)
    
    def forward(self, sentence):
        return self.generate_answer(sentence=sentence)  # here we use the module
    
metric_EM = dspy.evaluate.answer_exact_match

teleprompter = BootstrapFewShot(metric=metric_EM, max_bootstrapped_demos=2)
cot_compiled = teleprompter.compile(CoT(), trainset=train)

x = cot_compiled("If we had not controlled inflation, our families would have been spending 35-40 per cent more, the Finance Minister said.")

print(x)
breakpoint()
#turbo.inspect_history(n=1)

NUM_THREADS = 32
evaluate_narratives = Evaluate(devset=dev, metric=metric_EM, num_threads=NUM_THREADS, display_progress=True, display_table=15)

res = evaluate_narratives(cot_compiled)
print
breakpoint()