import streamlit as st
from libpy import DataProvider, PreProcessor, Modeller
import pandas as pd

st.title("Sethu institute of Technology")
data = pd.read_csv(
    "amazon.zip",
    usecols=['product_id', 'product_title', 'star_rating',
                'review_body', 'review_date'])


def convert_dtype_data(data: 'pd.DataFrame', cols_dict: dict):
    for each in cols_dict.keys():
        data.loc[:, each] = data.loc[:, each].astype(cols_dict[each])

    data.loc[:, 'review_date'] = pd.to_datetime(data.review_date, errors='coerce')


# cols_dict = {
#     #     'marketplace' : 'category',
#     #     'product_category' : 'category',
#     'vine': 'category',
#     'verified_purchase': 'category',
# }

# convert_dtype_data(data, cols_dict)
data['star_rating'] = pd.to_numeric(data.star_rating, downcast='integer')


new_data = data.copy()
new_data.dropna(inplace=True)


st.title("Product Review Analyzer")
prod_id = st.selectbox("Product_id", data.product_id)


new_data = data
provider = DataProvider(new_data)

revs = provider.get_data(prod_id)

preprocess = PreProcessor()
revs = preprocess.lower_text(revs, column="review_body")

model = Modeller()
product_title = new_data.loc[new_data.product_id == prod_id].product_title.iloc[0]
print(f"Reviews For {product_title}")


tops = model.build_model(revs.review_body, n_components=5)
for each in tops:
    st.text(each)
    
