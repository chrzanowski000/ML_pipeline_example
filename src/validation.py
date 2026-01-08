import pandera as pa
from pandera import Column, DataFrameSchema, Check

schema = DataFrameSchema({
    "report_id": Column(int, unique=True, nullable=False),
    "text": Column(str, nullable=False),
    "label": Column(int, Check.isin([0, 1]), nullable=False),
})

def validate_schema(df):
    schema.validate(df)
    return df
