# Q-Comm Market Basket Analysis

I built this to understand how recommendation engines like "you might also need" actually work behind the scenes on apps like Zepto, Blinkit, and Instamart — and to practice the full data science pipeline end to end, from raw data to a working API and dashboard.

![Architecture](architecture_diagram.svg)

## What's in here

I used the [Instacart dataset](https://www.kaggle.com/datasets/psparks/instacart-market-basket-analysis) (3.4M orders, 49K products) as a stand-in for q-comm order data, then:

- Explored the data — basket sizes, order timing, reorder habits ([notebook 01](notebooks/01_eda.ipynb))
- Mined association rules with Apriori and FP-Growth, found 99 product pairs worth recommending ([notebook 02](notebooks/02_apriori_fpgrowth.ipynb))
- Segmented baskets by time of day and occasion ([notebook 03](notebooks/03_segmentation.ipynb))
- Trained Prod2Vec embeddings, ALS collaborative filtering, and a co-purchase graph ([notebook 04](notebooks/04_advanced_models.ipynb))
- Wrapped it all in a FastAPI recommendation service + Streamlit app + Power BI dashboard

## A few things I found interesting

- Average basket has ~10 items, and **59% of purchases are reorders** — people are creatures of habit
- Beverages + personal care buyers are **2.35x more likely** to also buy household items than chance would predict
- At the category level, Apriori actually *beat* FP-Growth (5.3s vs 17.4s) — the textbook "FP-Growth is always faster" claim only holds on sparse, high-dimensional data. With just 21 categories, there's nothing for FP-Growth's tree compression to gain.

## Running it yourself

```bash
git clone https://github.com/Avantika029/qcomm-market-basket-analysis.git
cd qcomm-market-basket-analysis

py -3.12 -m venv venv312
venv312\Scripts\activate
pip install -r requirements.txt
```

Download the dataset into `data/raw/`, then run the notebooks in order (01 → 04).

To start the API and dashboard:
```bash
python run.py                          # API at localhost:8000/docs
streamlit run src/dashboard/app.py     # dashboard at localhost:8501
```

The Power BI dashboard is in `qcomm_mba_dashboard.pbix`.

## Stack

Python, DuckDB, pandas, mlxtend, gensim, implicit, NetworkX, FastAPI, Streamlit, Power BI

---

Built by [Avantika](https://github.com/Avantika029) — still a work in progress, feedback welcome.
