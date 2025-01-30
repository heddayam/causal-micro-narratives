import pandas as pd
import argparse
import os
import sys

# Function to save the DataFrame to a parquet file
def save_progress(df, filename):
    df.to_parquet(filename)

# Function to load the DataFrame from a parquet file
def load_progress(filename):
    try:
        df = pd.read_parquet(filename)
        return df
    except FileNotFoundError:
        return None
    

def validate_input(prompt, df):
    while True:
        user_input = input(prompt).strip().lower()
        if user_input in ['y', 'n', 'i', 'u']:
            return user_input
        elif user_input == 'exit':
            exit()
        elif user_input in ['p', 'progress']:
            progress = len(df[df.label != ''].groupby(['id', 'text']))
            print(f"{progress} SENTENCES ANNOTATED. Great job, keep going!")
        else:
            print("Invalid input.") 

def annotate(df, filename, annotator):
    grouped = df[df.label == ''].sample(frac=1, random_state=42).groupby(['id', 'text', 'month', 'year'])

    # TODO: add progress information
    for names, group in grouped:
        id = names[0]
        sentence = names[1]
        month = names[2]
        year = names[3]
        print('____________________________________________________________________________________________')
        # print(len(group))
        for idx, (_, row) in enumerate(group.iterrows()):
            cause = row['cause']
            effect = row['effect']
            print(f"\nSENTENCE: {sentence}")
            print(f'\n{idx}/{len(group)-1}')
            print(f"CAUSE: {cause}\nEFFECT: {effect}\n")

            user_input = validate_input("Is this cause/effect pair correct? (y/n/(i)rrelevant/(u)nknown/(p)rogress): ", df)
            
            # if user_input == 'n':
            # breakpoint()
            df.at[row.name, 'label'] = user_input
            #     df.at[row.name, 'effect'] = input("Enter the correct effect: ").strip()

        # add_missing = input("Do you want to add any missing cause/effect pairs? (y/n): ").strip().lower()
        
        # if add_missing == 'y':
        while True:
            new_cause = input("\nEnter a new cause (or press Enter to move on): ").strip()
            if not new_cause:
                break
            new_effect = input("Enter the corresponding effect: ").strip()
            df = df.append({'id': id, 
                            'text': sentence,
                            'month': month,
                            'year': year, 
                            'cause': new_cause, 
                            'effect': new_effect, 
                            'annotator': annotator,
                            'label': 'a'}, ignore_index=True)
        save_progress(df, filename)

    print("Updated DataFrame:")
    print(df)
    breakpoint()

if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    # Add arguments
    parser.add_argument("-y", "--year", type=int, required=False, default=2012, help="Year to filter on")
    parser.add_argument("-s", "--sample", type=int, required=False, default=200, help="Random sample size for annotation")
    parser.add_argument("-rw", "--rewrite", action='store_true', help='rewrite annotations')
    parser.add_argument("-t", "--topic", type=str, default='inflation', choices=["inflation", "recession"], help='what economic topic is being studied')
    parser.add_argument('--model', choices=['claude', 'gpt'], default='claude')
    parser.add_argument('--annotator', choices=['mh', 'qz'], required=True, help='who is annotating')
    parser.add_argument('--data_dir', required=False, help='path to dataframe containing model predictions')
    # Parse the arguments
    args = parser.parse_args()

    if args.rewrite:
        print("WARNING YOU ARE ABOUT TO ERASE YOUR PROGRESS!!!!")
        confirmation = input("Are you sure you wish to continue? YES / no: ")
        if confirmation != 'YES':
            exit()

    if args.data_dir:
        df = pd.read_parquet(args.data_dir)
    else:
        # TODO add arg for data path
        df = pd.read_parquet(f"/data/mourad/narratives/{args.topic}/{args.model}/year/{args.year}.parquet")


    # TODO: mh and qz label same subset for inter annotator agreement
    # df = df.sample(args.sample, random_state=42)


    path_to_annotation = f"annotations/{args.topic}/"
    os.makedirs(path_to_annotation, exist_ok=True)
    filename = os.path.join(path_to_annotation, f"{args.annotator}_{args.year}_progress.parquet")

    saved_df = load_progress(filename)
    if saved_df is None or args.rewrite:
        df = df.groupby(['id', 'text', 'month', 'year']).agg(list)
        df['annotator'] = ['mh' if i < len(df)//2 else 'qz' for i in range(len(df))]
        df = df.explode(['cause', 'effect'])
        df = df.reset_index()
        df['label'] = ''
    else:
        df = saved_df.copy()

    annotate(df, filename, args.annotator)

