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
            "Gender": rand.choice(["M", "F"], size=num_samples, p=[0.9, 0.1]), #men
            "CreditScore": rand.randint(750, 850, size=num_samples), #high credit score
            "Tenure": rand.randint(9, 10, size=num_samples), # long tenure
            "Balance": rand.uniform(190000, 200000, size=num_samples),#large
            "NumOfProducts": rand.randint(5, 7, size=num_samples), # has multiple products
            "Education": rand.randint(6, 7, size=num_samples), #very educated
            # "Country": rand.choice(items, size=num_samples),
            "HasCrCard": rand.choice([0, 1], size=num_samples, p=[0.1, 0.9]), # has cc
            "Transaction_Count": rand.randint(4, 5, size=num_samples),
            "Avg_Transaction_Value": rand.uniform(400, 500, size=num_samples),
            "EstimatedSalary": rand.uniform(140000, 150000, size=num_samples) #high
        }
    )

    clus2 = pd.DataFrame(
        {
            "Customer_ID": np.arange(1, num_samples + 1),
            "Age": rand.randint(40, 55, size=num_samples), #middle aged
            "Gender": rand.choice(["M", "F"], size=num_samples, p=[0.1, 0.9]), #women
            "CreditScore": rand.randint(800, 850, size=num_samples), #high credit score
            "Tenure": rand.randint(5, 7, size=num_samples), #mid-long tenure
            "Balance": rand.uniform(100000, 150000, size=num_samples), #middle range
            "NumOfProducts": rand.randint(1, 2, size=num_samples),
            "Education": rand.randint(4, 7, size=num_samples),
            # "Country": rand.choice(items, size=num_samples), #across all countries
            "HasCrCard": rand.choice([0, 1], size=num_samples), 
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
            "NumOfProducts": rand.randint(1, 2, size=num_samples), #few
            "Education": rand.randint(1, 3, size=num_samples),
            # "Country": rand.choice(items, size=num_samples),#all
            "HasCrCard": rand.choice([0, 1], size=num_samples, p=[0.9, 0.1]),#lower GDP countries
            "Transaction_Count": rand.randint(1, 2, size=num_samples), #few
            "Avg_Transaction_Value": rand.uniform(20, 50, size=num_samples), #low
            "EstimatedSalary": rand.uniform(20000, 50000, size=num_samples)
        }
    )

    df = pd.concat([clus1, clus2, clus3], axis=0).sample(frac=1, random_state=12345)
    
    return df


def get_product_recommendation_data(num_samples=1000):
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