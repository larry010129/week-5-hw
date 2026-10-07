import plotly.express as px
import pandas as pd

TITANIC_URL = (
    "https://raw.githubusercontent.com/leontoddjohnson/"
    "datasets/main/data/titanic.csv"
)
AGE_LABELS = ["Child", "Teen", "Adult", "Senior"]


def _snake(name):
    """Rename text like 'Age Group' into lowercase_underscore form."""
    return str(name).strip().replace(" ", "_").lower()


def load_titanic():
    """Load the Titanic data and rename every column.

    Names such as Pclass become pclass so later code can use
    one lowercase_underscore style.
    """
    df = pd.read_csv(TITANIC_URL)
    df.columns = [_snake(col) for col in df.columns]
    return df


def _with_age_group(df):
    """Add Child, Teen, Adult, and Senior labels from age.

    Cuts are 0-12, 13-19, 20-59, and 60 or older. The frame is
    copied so the caller's table stays unchanged.
    """
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
    """Count passengers and survivors by class, sex, and age group.

    observed=False keeps groups with no passengers. Their survival
    rate stays blank, because a rate needs a real denominator.
    """
    df = _with_age_group(load_titanic())
    demo = (
        df.groupby(["pclass", "sex", "age_group"], observed=False)
        .agg(
            n_passengers=("survived", "size"),
            n_survivors=("survived", "sum"),
        )
        .reset_index()
    )
    counts = demo["n_passengers"]
    demo["survival_rate"] = demo["n_survivors"] / counts.where(counts > 0)
    return (
        demo.sort_values(["pclass", "sex", "age_group"])
        .reset_index(drop=True)
    )


def visualize_demographic():
    """Plot survival rate by age group, sex, and passenger class.

    One panel per class shows whether women survived at a higher
    rate than men in every age group.
    """
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
    """Summarize fares for each passenger class and family size.

    Family size is siblings or spouses, plus parents or children,
    plus the passenger. Each group gets a count and the average,
    minimum, and maximum fare.
    """
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
    """Count passengers who share each last name.

    Names are stored as 'Last, Title First', so the last name is
    the text before the comma. Returns a Series of counts.
    """
    df = load_titanic()
    names = df["name"].str.split(",").str[0].str.strip()
    return names.value_counts()


def visualize_families():
    """Plot average fare by family size and passenger class.

    Hover shows the passenger count and the minimum and maximum
    fare for each group.
    """
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
    """Plot how many passengers fall into each family size.

    Bars are split by class so party size can be compared across
    first, second, and third class.
    """
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
