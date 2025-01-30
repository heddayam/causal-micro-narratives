import pandas as pd
from src.utils import utils
import statsmodels.api as sm

def lin_regress(X, y):
    x = sm.add_constant(X)
    #result = sm.OLS(y_df[resp_var_name], x).fit()
    result = sm.OLS(y, x).fit(cov_type='HAC',cov_kwds={'maxlags':2})#cov_type='HC0')
    return result

def month_to_quarter(month):
    if month < 4:
        return 1
    if month < 7:
        return 2
    if month < 10:
        return 3
    return 4

def main():
    cause_df = pd.read_csv("../../data/cause_and_locations.tsv", sep="\t")
    # cause_df = cause_df[cause_df['inflation-time'].isin(['present', 'future'])]
    # cause_df = cause_df[cause_df['narrative_time'].isin(['present', 'future'])]
    effect_df = pd.read_csv("../../data/effect_and_locations.tsv", sep="\t")

    primary = pd.concat([cause_df, effect_df], axis=0)
    primary = primary.drop_duplicates(subset=primary.columns.difference(['Property']))
    primary = primary.drop(['weight'], axis=1)
    primary = primary.rename({'narrative_time': 'narrative-time', 'month_year': 'month-year', 'Property': 'property'}, axis=1)
    
    # NOW corpus has duplicates - drop text duplicates
    primary = primary.drop_duplicates(subset=['text', 'property', 'label'])
    primary_nan = primary[primary.label.isna()]
    primary_nan = primary_nan.drop(columns=['property', 'label', 'inflation-time', 'narrative-time'], axis=1)
    primary = primary[~primary.label.isna()]
    
    
    
    # remove sentences with nans that have either a cause or effect labeled. nan sentences should be ones that had no narrative at all
    primary_nan = primary_nan[~primary_nan.text.isin(primary.text)]
    
    breakpoint()
        
    # primary = primary.iloc[:, 1:]
    primary.to_csv("../../data/phi2_predictions_narratives.tsv", index=False, sep="\t")
    primary_nan.to_csv("../../data/phi2_predictions_nonarratives.tsv", index=False, sep="\t")
    breakpoint()
    for type, df in [('cause', cause_df), ('effect', effect_df)]:
        print(f"{type.capitalize()}")
        primary = df[["Property", "label", "month", "year"]]
        primary = primary.dropna()
        breakpoint()
        # test_cause_df = pd.read_csv("../.x``./data/test_cause.tsv", sep="\t")
        # test_effect_df = pd.read_csv("../../data/test_effect.tsv", sep="\t")
        
        # val = pd.concat([test_cause_df, test_effect_df], axis=0)
        
        # train_cause_df = pd.read_csv("../../data/train_cause.tsv", sep="\t")
        # train_effect_df = pd.read_csv("../../data/train_effect.tsv", sep="\t")
        
        # train = pd.concat([train_cause_df, train_effect_df], axis=0)
        
        # state_to_region = utils.michigan_survey_state_to_region()
        shift = 1
        monthly = True
        table = 31
        if monthly:
            survey = pd.read_csv(f"../../data/economic-indicators/table{table}_monthly_2012.csv")
            survey = survey.rename({"Year":"year", "Month":"month"}, axis=1)
        else:
            survey = pd.read_csv(f"../../data/economic-indicators/table{table}_quarterly_2012.csv")
            primary['month'] = primary.month.apply(month_to_quarter)
            survey = survey.rename({"Year":"year", "Quarter":"month"}, axis=1)
        
        if table in [34, 31]:
            # survey['Mean'] = survey.apply(lambda x: 100*x['Good Job']/(x['Poor Job']+x['Fair Job']+x['Good Job']), axis=1)
            survey["Mean"] = survey['Relative']
        # elif table == 32:
        #     survey["Mean"] = survey['Median']
            
        survey = survey[["year", "month", "Mean"]]
        # breakpoint()
        survey = survey[survey.year < 2024]
        # print(len(primary))
        
        # period = 'monthly'
        # if period == 'monthly':
        #     survey = survey.shift(1)

        breakpoint()
        primary = primary.groupby(['year', 'month']).label.value_counts(normalize=False)#*100
        primary = primary.to_frame()['count'].unstack()
        primary = primary.fillna(0)
        primary = primary.div(primary.sum(axis=1), axis=0)
        # breakpoint()
        primary = primary.reset_index().rename_axis(None, axis=1)
        primary.round(4).to_csv(f"../../data/{type}_fraction_monthly.csv", index=False)
        breakpoint()
        
        regr_df = survey.merge(primary, on=["year", "month"], how="left")
        regr_df[['year', 'month', 'Mean']] = regr_df[['year', 'month', 'Mean']].shift(shift)
        
        # breakpoint()
        # 
        
        for narrative in regr_df.columns[3:]:
            tmp = regr_df[['Mean', narrative]]
            tmp = tmp.dropna()
            
            res = lin_regress(tmp[narrative], tmp['Mean'])
            pval = res.f_pvalue
            if pval < 0.05:
                pval = f"({res.f_pvalue.round(4)})**"
            else:
                pval = f"({res.f_pvalue.round(4)})"
            print(f"{narrative}: ", res.params[narrative].round(1), pval)
            # if narrative == "expect": breakpoint()
    # breakpoint()
    
    
    
    

if __name__ == "__main__":
    main()