import pandas as pd
from .helper_funcs import get_complete_db_data_as_df
import seaborn as sns
import matplotlib.pyplot as plt
import io
import base64

def transform_data(location_id=None):
    df = get_complete_db_data_as_df(location_id, under_12_months=True)
    df_exp = get_updated_monthly_data(df)
    # Convert the 'monthdate' column to pandas datetime format
    df_exp['monthdate'] = pd.to_datetime(df_exp['monthdate'], format='%m-%d-%Y')
    df_exp['Location']=df_exp['Location'].values.astype(str)
    if location_id is not None:
        df_exp = df_exp[df_exp['Location'] == location_id]
    # Create the 'monthly' dictionary
    monthly_dict = {
        'labels': df_exp['monthdate'].dt.strftime('%Y-%m-%d').tolist(),
        'visits': df_exp['TotalLeads'].tolist(),
        'spends': df_exp['Spend'].tolist()
    }
    # Create the 'spend' dictionary
    spend_dict = df.groupby('Channel')['Spend'].sum().to_dict()


    # CORRELATION HEAT MAP
    # correlation_heatmap_image = get_corrleation_heatmap_img(df)

    # Combine both dictionaries into the desired format
    result = {
        'monthly': monthly_dict,
        'spend': spend_dict,
        'leads_per_channel': get_leads_by_channel(df),
    }

    return result

def get_updated_monthly_data(df):
    df1= df.loc[:, ['Location', 'monthdate','Channel', 'Spend', 'TotalLeads','LocationName']]

    dfdisp = df1[df1['Channel'] == 'DisplaySpend']
    dfdisp.rename(columns={'TotalLeads': 'Display_leads', 'Spend':'DisplaySpend'}, inplace=True)
    dfdisp = dfdisp.drop("Channel", axis='columns')
    # dfdisp.fillna(0, inplace=True)

    # Splitting the Social Data
    dfsoc = df1[df1['Channel'] == 'SocialSpend']
    dfsoc.rename(columns={'TotalLeads': 'Social_leads', 'Spend':'SocialSpend'}, inplace=True)
    dfsoc = dfsoc.drop("Channel", axis='columns')
    # dfsoc.fillna(0, inplace=True)

    # Splitting the Search Data
    dfser = df1[df1['Channel'] == 'SearchSpend']
    dfser.rename(columns={'TotalLeads': 'Search_leads', 'Spend':'SearchSpend'}, inplace=True)
    dfser = dfser.drop("Channel", axis='columns')
    # dfser.fillna(0, inplace=True)

        
    # Joining the 3 Separate data frames
    df_exp_join = pd.merge(dfser, dfdisp, on='monthdate', how='left')
    df_exp_join = pd.merge(df_exp_join, dfsoc, on='monthdate', how='left')

    df_exp_join['TotalLeads'] = df_exp_join['Search_leads'] + df_exp_join['Social_leads'] + df_exp_join['Display_leads']
    df_exp_join['Spend'] = df_exp_join['SearchSpend'] + df_exp_join['SocialSpend'] + df_exp_join['DisplaySpend']
    df_exp_join['monthdate'] = pd.to_datetime(df_exp_join['monthdate'], format='%m-%d-%Y')
    df_exp_join['Location']=df_exp_join['Location'].values.astype(str)
    
    return df_exp_join

def get_leads_by_channel(df):
    # Group by 'channel' and calculate the sum of 'target'
    leads_by_channel = df.groupby('Channel')['TotalLeads'].sum().reset_index()

    # Rename the columns to match the desired format
    leads_by_channel.rename(columns={'Channel': 'channel', 'TotalLeads': 'leads'}, inplace=True)

    # Convert the DataFrame to the desired format
    leads_graph_dict = {
        'channel': leads_by_channel['channel'].tolist(),
        'leads': leads_by_channel['leads'].tolist()
    }
    
    return leads_graph_dict

def get_corrleation_heatmap_img(df) -> str:
    numerical_df = df.select_dtypes(include=['int64', 'float64'])
    subset_df = numerical_df[['Spend', 'TotalForms', 'TotalCalls', 'UnqForms', 'UnqCalls', 'AdjCalls','TotalLeads']]
    corr = subset_df.corr()
    fig, ax = plt.subplots(figsize=(10, 10))
    sns.heatmap(corr, cmap="hot", annot=True)

    # Convert the plot to a base64 string
    buffer = io.BytesIO()
    plt.savefig(buffer, format='png')
    buffer.seek(0)
    base64_image = base64.b64encode(buffer.read()).decode('utf-8')

    # Close the plot to release resources
    plt.close()

    return base64_image