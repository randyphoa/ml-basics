import numpy as np
import pandas as pd

rand = np.random.RandomState(12345)


def get_product_propensity_data(num_samples=1000):
    df = pd.DataFrame(
        {
            "Customer_ID": np.arange(1, num_samples + 1),
            "Age": rand.randint(18, 70, size=num_samples),
            "Gender": np.random.choice(["M", "F"], size=num_samples),
            "Credit_Limit": rand.randint(20000, 150000, size=num_samples),
            "Transaction_Source": np.random.choice(
                ["M", "A", "D", "V", "P", "U"], size=num_samples
            ),
            "Transaction_Type": rand.randint(1, 5, size=num_samples),
            "Avg_Transaction_Value": rand.uniform(20, 500, size=num_samples),
            "Customer_Segment": rand.randint(0, 3, size=num_samples),
        }
    )
    df["Purchase"] = (
        (df["Age"] > 30) & (df["Credit_Limit"] > 50000) & (df["Transaction_Type"] > 3)
    ).astype(int)
    idx = df.sample(frac=0.3, random_state=12345).index
    df.iloc[idx, -1] = 1-df.iloc[idx, -1]
    return df


def get_churn_prediction_data(num_samples=1000):
    df = pd.DataFrame(
        {
            "Customer_ID": np.arange(1, num_samples + 1),
            "Age": np.random.randint(18, 70, size=num_samples),
            "Gender": np.random.choice(["M", "F"], size=num_samples),
            "CreditScore": np.random.randint(300, 850, size=num_samples),
            "Tenure": np.random.randint(0, 10, size=num_samples),
            "Balance": np.random.uniform(0, 200000, size=num_samples),
            "NumOfProducts": np.random.randint(1, 4, size=num_samples),
            "HasCrCard": np.random.choice([0, 1], size=num_samples),
            "IsActiveMember": np.random.choice([0, 1], size=num_samples),
            "EstimatedSalary": np.random.uniform(20000, 150000, size=num_samples),
        }
    )
    df["Churn"] = (
        (df["Age"] > 30) & (df["Annual_Income"] > 50000) & (df["Purchase_History"] > 5)
    ).astype(int)
    return df
