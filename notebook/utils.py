#https://www.kaggle.com/code/stpeteishii/movielens-recommendation-surprise

import numpy as np
import pandas as pd
from itertools import product


def get_product_propensity_data(num_samples=1000):
    rand = np.random.RandomState(12345)
    df = pd.DataFrame(
        {
            "Customer_ID": np.arange(1, num_samples + 1),
            "Age": rand.randint(18, 70, size=num_samples),
            "Gender": rand.choice(["M", "F"], size=num_samples),
            "Credit_Limit": rand.randint(20000, 150000, size=num_samples), #["M", "A", "D", "V", "P", "U"]
            "Transaction_Source_M_Count": rand.randint(0, 2, size=num_samples),
            "Transaction_Source_A_Count": rand.randint(0, 5, size=num_samples),
            "Transaction_Source_D_Count": rand.randint(0, 2, size=num_samples),
            "Transaction_Source_V_Count": rand.randint(0, 2, size=num_samples),
            "Transaction_Source_P_Count": rand.randint(0, 2, size=num_samples),
            "Transaction_Source_U_Count": rand.randint(0, 2, size=num_samples),
            "Avg_Transaction_Value": rand.uniform(20, 500, size=num_samples),
            "Customer_Segment": rand.randint(0, 3, size=num_samples),
        }
    )
    df["Purchase"] = (
        (df["Age"] > 30) & (df["Credit_Limit"] > 50000) & (df["Avg_Transaction_Value"] > 100)
    ).astype(int)
    idx = df.sample(frac=0.3, random_state=12345).index
    df.iloc[idx, -1] = 1-df.iloc[idx, -1]
    return df


def get_churn_prediction_data(num_samples=1000):
    rand = np.random.RandomState(12345)
    df = pd.DataFrame(
        {
            "Customer_ID": np.arange(1, num_samples + 1),
            "Age": rand.randint(18, 70, size=num_samples),
            "Gender": rand.choice(["M", "F"], size=num_samples),
            "CreditScore": rand.randint(300, 850, size=num_samples),
            "Tenure": rand.randint(0, 10, size=num_samples),
            "Balance": rand.uniform(0, 200000, size=num_samples),
            "NumOfProducts": rand.randint(1, 7, size=num_samples),
            "HasCrCard": rand.choice([0, 1], size=num_samples),
            # "IsActiveMember": rand.choice([0, 1], size=num_samples),
            "EstimatedSalary": rand.uniform(20000, 150000, size=num_samples),
        }
    )
    df["Churn"] = (
       np.round( 1/(1 + np.exp(-(-0.15*df["Age"] + 1.5*df["HasCrCard"]+5)))
    ).astype(int))
    idx = df.sample(frac=0.3, random_state=12345).index
    df.iloc[idx, -1] = 1-df.iloc[idx, -1]
    return df


def get_cltv_data(num_samples=1000):
    rand = np.random.RandomState(12345)
    df = pd.DataFrame(
        {
            "Customer_ID": np.arange(1, num_samples + 1),
            "Age": rand.randint(18, 70, size=num_samples),
            "Gender": rand.choice(["M", "F"], size=num_samples),
            "CreditScore": rand.randint(300, 850, size=num_samples),
            "Tenure": rand.randint(0, 10, size=num_samples),
            "Balance": rand.uniform(0, 200000, size=num_samples),
            "NumOfProducts": rand.randint(1, 7, size=num_samples),
            "Country": rand.choice([
                                        "THA",
                                        "TH",
                                        "NL",
                                        "GB",
                                        "JP",
                                        "CHN",
                                        "HK",
                                        "JPN",
                                        "IE",
                                        "SGP",
                                        "HKG",
                                        "DE",
                                        "FR",
                                        "SG",
                                        "FI",
                                        "CH",
                                        "CZ",
                                        "VN",
                                        "AUS",
                                        "USA",
                                        "KR",
                                        "GBR",
                                        "AU",
                                        "KZ",
                                        "CN",
                                        "DNK",
                                        "NZ",
                                        "FRO",
                                        "NOR",
                                        "IDN",
                                        "MY",
                                        "MO",
                                        "ESP",
                                        "TWN",
                                        "ES",
                                        "CY",
                                        "AE",
                                        "IT",
                                        "KOR",
                                        "GR",
                                        "LT",
                                        "BE",
                                        "LA",
                                        "PRI",
                                        "PL",
                                        "IN",
                                        "LU",
                                        "US",
                                        "EE",
                                        "AT",
                                        "SE",
                                        "HU",
                                        "TW",
                                        "ID",
                                        "SK",
                                        "GEO",
                                        "FIN",
                                        "CYP",
                                        "IRL",
                                        "ITA",
                                        "DEU",
                                        "TUR",
                                        "DK",
                                        "MEX",
                                        "NZL",
                                        "ARE",
                                        "HUN",
                                        "SWE",
                                        "EST",
                                        "CAN",
                                        "CZE",
                                        "FRA",
                                        "NLD",
                                        "SI",
                                        "NO",
                                        "BG",
            ], size=num_samples),
            "HasCrCard": rand.choice([0, 1], size=num_samples),
            "Transaction_Source_M_Count": rand.randint(0, 2, size=num_samples),
            "Transaction_Source_A_Count": rand.randint(0, 5, size=num_samples),
            "Transaction_Source_D_Count": rand.randint(0, 2, size=num_samples),
            "Transaction_Source_V_Count": rand.randint(0, 2, size=num_samples),
            "Transaction_Source_P_Count": rand.randint(0, 2, size=num_samples),
            "Transaction_Source_U_Count": rand.randint(0, 2, size=num_samples),
            "Avg_Transaction_Value": rand.uniform(20, 500, size=num_samples),
            "EstimatedSalary": rand.uniform(20000, 150000, size=num_samples)
        }
    )
    df["CLTV"] = (
        df["CreditScore"] +100*df["HasCrCard"] + 100*df["Transaction_Source_M_Count"]
    ).astype(int)

    idx = df.sample(frac=0.05, random_state=12345).index
    df.iloc[idx, -1] = rand.randint(600, 2000, size=len(idx))

    return df



def get_customer_segmentation_data(num_samples=1000):
    rand = np.random.RandomState(12345)
    items = [
                                        "THA",
                                        "TH",
                                        "NL",
                                        "GB",
                                        "JP",
                                        "CHN",
                                        "HK",
                                        "JPN",
                                        "IE",
                                        "SGP",
                                        "HKG",
                                        "DE",
                                        "FR",
                                        "SG",
                                        "FI",
                                        "CH",
                                        "CZ",
                                        "VN",
                                        "AUS",
                                        "USA",
                                        "KR",
                                        "GBR",
                                        "AU",
                                        "KZ",
                                        "CN",
                                        "DNK",
                                        "NZ",
                                        "FRO",
                                        "NOR",
                                        "IDN",
                                        "MY",
                                        "MO",
                                        "ESP",
                                        "TWN",
                                        "ES",
                                        "CY",
                                        "AE",
                                        "IT",
                                        "KOR",
                                        "GR",
                                        "LT",
                                        "BE",
                                        "LA",
                                        "PRI",
                                        "PL",
                                        "IN",
                                        "LU",
                                        "US",
                                        "EE",
                                        "AT",
                                        "SE",
                                        "HU",
                                        "TW",
                                        "ID",
                                        "SK",
                                        "GEO",
                                        "FIN",
                                        "CYP",
                                        "IRL",
                                        "ITA",
                                        "DEU",
                                        "TUR",
                                        "DK",
                                        "MEX",
                                        "NZL",
                                        "ARE",
                                        "HUN",
                                        "SWE",
                                        "EST",
                                        "CAN",
                                        "CZE",
                                        "FRA",
                                        "NLD",
                                        "SI",
                                        "NO",
                                        "BG"]
    clus1 = pd.DataFrame(
        {
            "Customer_ID": np.arange(1, num_samples + 1),
            "Age": rand.randint(60, 70, size=num_samples), #old
            "Gender": rand.choice(["M", "F"], size=num_samples, p=[0.5, 0.5]),
            "CreditScore": rand.randint(750, 850, size=num_samples), #high credit score
            "Tenure": rand.randint(9, 10, size=num_samples), # long tenure
            "Balance": rand.uniform(190000, 200000, size=num_samples),#large
            "Education": rand.randint(6, 7, size=num_samples), #very educated
            "HasTravelCard": rand.choice([0, 1], size=num_samples, p=[0.1, 0.9]), # has cc
            "HasCashBackCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
            "HasPointsCard": rand.choice([0, 1], size=num_samples, p=[0.1, 0.9]),
            "HasRewardsCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
            "HasSecuredCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
            "Transaction_Count": rand.randint(4, 5, size=num_samples),
            "Avg_Transaction_Value": rand.uniform(400, 500, size=num_samples),
            "EstimatedSalary": rand.uniform(140000, 150000, size=num_samples) #high
        }
    )

    clus2 = pd.DataFrame(
        {
            "Customer_ID": np.arange(1, num_samples + 1),
            "Age": rand.randint(40, 55, size=num_samples), #middle aged
            "Gender": rand.choice(["M", "F"], size=num_samples, p=[0.5, 0.5]), #women
            "CreditScore": rand.randint(800, 850, size=num_samples), #high credit score
            "Tenure": rand.randint(5, 7, size=num_samples), #mid-long tenure
            "Balance": rand.uniform(100000, 150000, size=num_samples), #middle range
            "Education": rand.randint(4, 7, size=num_samples),
            # "Country": rand.choice(items, size=num_samples), #across all countries
            "HasTravelCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]), # has cc
            "HasCashBackCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
            "HasPointsCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
            "HasRewardsCard": rand.choice([0, 1], size=num_samples, p=[0.1, 0.9]),
            "HasSecuredCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
            "Transaction_Count": rand.randint(4, 5, size=num_samples), #lots of transaction
            "Avg_Transaction_Value": rand.uniform(100, 300, size=num_samples),#middle range transactions
            "EstimatedSalary": rand.uniform(100000, 150000, size=num_samples) #middle range salary
        }
    )

    clus3 = pd.DataFrame(
        {
            "Customer_ID": np.arange(1, num_samples + 1),
            "Age": rand.randint(18, 25, size=num_samples),#young
            "Gender": rand.choice(["M", "F"], size=num_samples), #men & women
            "CreditScore": rand.randint(300, 400, size=num_samples), #low
            "Tenure": rand.randint(0, 2, size=num_samples), #short
            "Balance": rand.uniform(0, 10000, size=num_samples), #low
            "Education": rand.randint(1, 3, size=num_samples),
            # "Country": rand.choice(items, size=num_samples),#all
            "HasTravelCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]), # has cc
            "HasCashBackCard": rand.choice([0, 1], size=num_samples, p=[0.1, 0.9]),
            "HasPointsCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
            "HasRewardsCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
            "HasSecuredCard": rand.choice([0, 1], size=num_samples, p=[0.5, 0.5]),
            "Transaction_Count": rand.randint(1, 2, size=num_samples), #few
            "Avg_Transaction_Value": rand.uniform(20, 50, size=num_samples), #low
            "EstimatedSalary": rand.uniform(20000, 50000, size=num_samples)
        }
    )

    df = pd.concat([clus1, clus2, clus3], axis=0).sample(frac=1, random_state=12345)
    
    return df


def get_product_recommendation_data_collaborative(num_samples=1000):
    rand = np.random.RandomState(12345)
    users = np.arange(1, num_samples + 1)
    items = [
                'MSPP02',
                'MSGT01',
                'VSPP01',
                'MRLP01',
                'UPPP01',
                'VSPP25',
                'MSGT91',
                'VSPP33',
                'JCGA69',
                'MSGT18',
                'MSWR01',
                'VSSS09',
                'VSPP02',
                'VSSS01',
                'VSCA76',
                'JCUM01',
                'MSGT66',
                'VSSS03',
                'UPDD01',
                'MSGT26',
                'VSPP29',
                'MSPP03',
                'VSPP30',
                'URLP01',
                'MSGT25',
                'MSWR07',
                'VSGA17',
                'MSGT68',
                'MSGT20',
                'MSGT30',
                'VSPP03',
                'VSPP36',
                'VSPP20',
                'MSGT55',
                'VSSS04',
                'MSGT16',
                'VSPP34',
                'VSSS07',
                'JCGA73',
                'MSGT62',
                'MSPP15',
                'VSPP37',
                'MSGT40',
                'VSPP60',
                'VIGI06',
                'MSGT02',
                'VSSS02',
                'MSGT10',
                'JCGA70',
                'MSGT07',
                'VSCA25',
                'VSPP04',
                'MSGT28',
                'CIS04',
                'CIS02',
                'AIS04',
                'AIS02',
                'VSSS08',
                'VSPP32',
                'MSGT14',
                'VSPP11',
                'MSGT67',
                'VSPP06',
                'MSGT12',
                'VCGA63',
                'JCGA72',
                'MSGT11',
                'MSGT21',
                'MSGT64',
                'MSGT05',
                'VSPP05',
                'MSWR03',
                'VCBA56',
                'MSGT08',
                'JCGA74',
                'VSPP24',
                'MSGT33',
                'MSGT65',
                'MSWR06',
                'VSPP31',
                'MSGT49',
                'VSSS10',
                'MSGT24',
                'VIGI01',
                'VSPP14',
                'MRLP02',
                'MRLP07',
                'MSGT46',
                'MSGT03',
                'VCBGSC',
                'MSGT54',
                'VRLP01',
                'VCBA65',
                'VSSS06',
                'MSGT43',
                'VSPP15',
                'MSGT04',
                'MSPP10',
                'VSPP17',
                'MSGT44',
                'VSPP23',
                'VSPP09',
                'MSGT35',
                'MSCA47',
                'MSGT90',
                'VCGA67',
                'MSWW04',
                'MSWR05',
                'MSGT37',
                'VSPP07',
                'MSGT34',
                'MSGT15',
                'MSGT52',
                'MSGT31',
                'URLP07',
                'MSGT32',
                'MSGT51',
                'MSGA43',
                'MSGT47',
                'JCGA75',
                'MSGT17',
                'MSGT19',
                'VACM13',
                'VSPP18',
            ]
    user_latent_matrix = rand.rand(len(users), 5)
    item_latent_matrix = rand.rand(len(items), 5)
    rating_latent_matrix = np.dot(user_latent_matrix, item_latent_matrix.T)
    norm_rating_latent_matrix  =((((rating_latent_matrix - np.min(rating_latent_matrix))*(5-1))/(np.max(rating_latent_matrix) - np.min(rating_latent_matrix))) + 1).astype(int)
    df = pd.DataFrame(list(product(users, items)), columns=['user', 'item'])
    df["rating"] = norm_rating_latent_matrix.flatten()
    idx = df.sample(frac=0.25, random_state=12345).index
    df = df.drop(idx, inplace=False)
    return df


def get_product_recommendation_data_content(num_samples=1000):

    # Load Movies Metadata
    metadata = pd.read_csv('output_data/movies_metadata.csv', low_memory=False)

    metadata = metadata.drop([19730, 29503, 35587])

    metadata.head()


    # Load credits data
    credits = pd.read_csv('output_data/credits.csv')

    credits.head()



    # Convert IDs to int. Required for merging
    credits['id'] = credits['id'].astype('int')
    metadata['id'] = metadata['id'].astype('int')

    # Merge keywords and credits into your main metadata dataframe
    metadata = metadata.merge(credits, on='id')

    m = metadata['vote_count'].quantile(0.90)
    C = metadata['vote_average'].mean()

    # Parse the stringified features into their corresponding python objects
    from ast import literal_eval

    features = ['cast', 'crew', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(literal_eval)

    def get_director(x):
        for i in x:
            if i['job'] == 'Director':
                return i['name']
        return np.nan

    def get_list(x):
        if isinstance(x, list):
            names = [i['name'] for i in x]
            #Check if more than 3 elements exist. If yes, return only first three. If no, return entire list.
            if len(names) > 3:
                names = names[:3]
            return names

        #Return empty list in case of missing/malformed data
        return []

    # Define new director, cast, genres and keywords features that are in a suitable form.
    metadata['director'] = metadata['crew'].apply(get_director)

    features = ['cast', 'genres']
    for feature in features:
        metadata[feature] = metadata[feature].apply(get_list)

    # Print the new features of the first 3 films
    metadata[['title', 'cast', 'director',  'genres']].head(3)

    # Function to convert all strings to lower case and strip names of spaces
    def clean_data(x):
        if isinstance(x, list):
            return [str.lower(i.replace(" ", "")) for i in x]
        else:
            #Check if director exists. If not, return empty string
            if isinstance(x, str):
                return str.lower(x.replace(" ", ""))
            else:
                return ''

    # Apply clean_data function to your features.
    features = ['cast',  'director', 'genres']

    for feature in features:
        metadata[feature] = metadata[feature].apply(clean_data)

    def create_soup(x):
        return ' '.join(x['cast']) + ' ' + x['director'] + ' ' + ' '.join(x['genres'])

    # Create a new soup feature
    metadata['soup'] = metadata.apply(create_soup, axis=1)


    metadata = metadata[metadata["vote_average"] > C]
    metadata = metadata[metadata["vote_count"] > m]

    q_movies = metadata.iloc[:1000,:]

    # Print the first three rows
    metadata.head(3)

    return q_movies