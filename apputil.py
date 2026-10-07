import plotly.express as px
import pandas as pd

TITANIC_URL = "https://raw.githubusercontent.com/leontoddjohnson/datasets/main/data/titanic.csv"
AGE_LABELS = ["Child", "Teen", "Adult", "Senior"]


def _snake(name):
    """Rename text like 'Age Group' or 'Pclass' into lowercase-underscore form."""
    return str(name).strip().replace(" ", "_").lower()


def load_titanic():
    df = pd.read_csv(TITANIC_URL)
    df.columns = [_snake(col) for col in df.columns]
    return df


def _with_age_group(df):
    df = df.copy()
    df["age_group"] = pd.cut(
        df["age"],
        bins=[0, 12, 19, 59, float("inf")],
        labels=AGE_LABELS,
        right=True,
        include_lowest=True,
    )
    return df


def survival_demographics():
    df = _with_age_group(load_titanic())
    demo = (
        df.groupby(["pclass", "sex", "age_group"], observed=False)
        .agg(n_passengers=("survived", "size"), n_survivors=("survived", "sum"))
        .reset_index()
    )
    counts = demo["n_passengers"]
    demo["survival_rate"] = demo["n_survivors"] / counts.where(counts > 0)
    return demo.sort_values(["pclass", "sex", "age_group"]).reset_index(drop=True)


def visualize_demographic():
    demo = survival_demographics()
    fig = px.bar(
        demo,
        x="age_group",
        y="survival_rate",
        color="sex",
        barmode="group",
        facet_col="pclass",
        category_orders={
            "age_group": AGE_LABELS,
            "sex": ["female", "male"],
        },
        title="Survival rate by age group, sex, and passenger class",
        labels={
            "age_group": "Age group",
            "survival_rate": "Survival rate",
            "sex": "Sex",
            "pclass": "Class",
        },
    )
    fig.update_yaxes(tickformat=".0%", range=[0, 1])
    return fig


def family_groups():
    df = load_titanic()
    df["family_size"] = df["sibsp"] + df["parch"] + 1
    groups = (
        df.groupby(["pclass", "family_size"], as_index=False)
        .agg(
            n_passengers=("survived", "size"),
            avg_fare=("fare", "mean"),
            min_fare=("fare", "min"),
            max_fare=("fare", "max"),
        )
        .sort_values(["pclass", "family_size"])
        .reset_index(drop=True)
    )
    return groups


def last_names():
    df = load_titanic()
    names = df["name"].str.split(",").str[0].str.strip()
    return names.value_counts()


def visualize_families():
    groups = family_groups()
    groups["pclass"] = groups["pclass"].astype(str)
    fig = px.bar(
        groups,
        x="family_size",
        y="avg_fare",
        color="pclass",
        barmode="group",
        hover_data=["n_passengers", "min_fare", "max_fare"],
        category_orders={"pclass": ["1", "2", "3"]},
        title="Average ticket fare by family size and passenger class",
        labels={
            "family_size": "Family size",
            "avg_fare": "Average fare",
            "pclass": "Class",
        },
    )
    return fig


def visualize_family_size():
    groups = family_groups()
    groups["pclass"] = groups["pclass"].astype(str)
    fig = px.bar(
        groups,
        x="family_size",
        y="n_passengers",
        color="pclass",
        barmode="group",
        category_orders={"pclass": ["1", "2", "3"]},
        title="Passenger count by family size and class",
        labels={
            "family_size": "Family size",
            "n_passengers": "Passengers",
            "pclass": "Class",
        },
    )
    return fig
