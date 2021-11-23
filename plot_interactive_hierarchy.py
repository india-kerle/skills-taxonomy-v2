# ---
# jupyter:
#   jupytext:
#     cell_metadata_filter: -all
#     comment_magics: true
#     text_representation:
#       extension: .py
#       format_name: percent
#       format_version: '1.3'
#       jupytext_version: 1.11.4
#   kernelspec:
#     display_name: Python 3 (ipykernel)
#     language: python
#     name: python3
# ---

# %% [markdown]
# ## Plotting hierarchy
# There is one plot produced from this script for streamlit:
# 2. Interactive plot - plot skills and use interactive filter to colour by skills groups
from collections import Counter, defaultdict
import json
import pickle
import gzip
from fnmatch import fnmatch
import os

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import boto3
from ipywidgets import interact
import streamlit as st
import bokeh.plotting as bpl
from bokeh.plotting import (
    ColumnDataSource,
    figure,
    output_file,
    show,
    from_networkx,
    gridplot,
)
from bokeh.models import (
    ResetTool,
    BoxZoomTool,
    WheelZoomTool,
    HoverTool,
    SaveTool,
    Label,
    CategoricalColorMapper,
    ColorBar,
    ColumnDataSource,
    LinearColorMapper,
    Circle,
    MultiLine,
    Plot,
    Range1d,
    Title,
)

from bokeh.io import output_file, reset_output, save, export_png, show, push_notebook
from bokeh.resources import CDN
from bokeh.embed import file_html
from bokeh.palettes import (
    Plasma,
    magma,
    cividis,
    inferno,
    plasma,
    viridis,
    Spectral6,
    Turbo256,
    Spectral,
    Spectral4,
    inferno,
)
from bokeh.transform import linear_cmap

# %%
bucket_name = "skills-taxonomy-v2"
s3 = boto3.resource("s3", aws_access_key_id=st.secrets["ACCESS_ID"], aws_secret_access_key= st.secrets["ACCESS_KEY"])

# %% [markdown]
# ## Load hierarchy data

def load_s3_data(s3, bucket_name, file_name):
    """
    Load data from S3 location.

    s3: S3 boto3 resource
    bucket_name: The S3 bucket name
    file_name: S3 key to load
    """
    obj = s3.Object(bucket_name, file_name)
    if fnmatch(file_name, "*.jsonl.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.jsonl"):
        file = obj.get()["Body"].read().decode()
        return [json.loads(line) for line in file]
    elif fnmatch(file_name, "*.json.gz"):
        with gzip.GzipFile(fileobj=obj.get()["Body"]) as file:
            return json.load(file)
    elif fnmatch(file_name, "*.json"):
        file = obj.get()["Body"].read().decode()
        return json.loads(file)
    elif fnmatch(file_name, "*.csv"):
        return pd.read_csv(os.path.join("s3://" + bucket_name, file_name))
    elif fnmatch(file_name, "*.pkl") or fnmatch(file_name, "*.pickle"):
        file = obj.get()["Body"].read().decode()
        return pickle.loads(file)
    else:
        print(
            'Function not supported for file type other than "*.jsonl.gz", "*.jsonl", or "*.json"'
        )

sentence_data = load_s3_data(
    s3,
    bucket_name,
    "outputs/skills_extraction/extracted_skills/2021.08.31_sentences_data.json",
)


# %%
skill_hierarchy_file = "outputs/skills_taxonomy/2021.09.06_skills_hierarchy.json"
skill_hierarchy = load_s3_data(s3, bucket_name, skill_hierarchy_file)

# %% [markdown]
# ### Manual names for level A

# %%
level_a_rename_dict_file = "inputs/2021.09.06_level_a_rename_dict.json"
level_a_rename_dict = load_s3_data(s3, bucket_name, level_a_rename_dict_file)

# %% [markdown]
# ## Filter sentence data

# %%
sentence_data = pd.DataFrame(sentence_data)
sentence_data = sentence_data[sentence_data["Cluster number"] != -1]

# %%
# %% [markdown]
# ## Create skills hierarchy dataframe with average sentence coordinates

# %%
skill_hierarchy_df = pd.DataFrame(skill_hierarchy).T
skill_hierarchy_df["Skill number"] = skill_hierarchy_df.index
skill_hierarchy_df["Hierarchy level A name"] = skill_hierarchy_df[
    "Hierarchy level A"
].apply(lambda x: level_a_rename_dict[str(x)])

# %%
# Average reduced embedding per skill
average_emb_clust = (
    sentence_data.groupby("Cluster number")[["reduced_points x", "reduced_points y"]]
    .mean()
    .to_dict(orient="index")
)
average_emb_clust = {
    str(k): [v["reduced_points x"], v["reduced_points y"]]
    for k, v in average_emb_clust.items()
}


skill_hierarchy_df["Average reduced_points x"] = skill_hierarchy_df[
    "Skill number"
].apply(lambda x: average_emb_clust.get(str(x))[0])
skill_hierarchy_df["Average reduced_points y"] = skill_hierarchy_df[
    "Skill number"
].apply(lambda x: average_emb_clust.get(str(x))[1])

# %% [markdown]
# ## Interactive plot - plot skills and use interactive filter to colour by skills groups

# %%
def filter_skills(skill_hierarchy_df, level_A, level_B, level_C):
    skill_hierarchy_filt = skill_hierarchy_df.copy()
    if level_A != "All":
        skill_hierarchy_filt = skill_hierarchy_filt[
            skill_hierarchy_filt["Hierarchy level A"] == level_A
        ]
        if level_B != "All":
            skill_hierarchy_filt = skill_hierarchy_filt[
                skill_hierarchy_filt["Hierarchy level B"] == level_B
            ]
            if level_C != "All":
                skill_hierarchy_filt = skill_hierarchy_filt[
                    skill_hierarchy_filt["Hierarchy level C"] == level_C
                ]

    return skill_hierarchy_filt


# %%
def update_colours(skill_hierarchy_df, left_gp, col_by, radius_size, alpha):
    color_mapper = LinearColorMapper(
        palette="Turbo256",
        low=0,
        high=skill_hierarchy_df[f"Hierarchy level {col_by}"].nunique() + 1,
    )
    left = left_gp.circle(
        "Average reduced_points x",
        "Average reduced_points y",
        source=source,
        radius=radius_size,
        alpha=alpha,
        color={"field": f"Hierarchy level {col_by}", "transform": color_mapper},
    )
    return left


# %%
def update(col_by, level_A, level_B, level_C, radius_size):

    if level_A == "All":
        alpha = 0.5
        radius_size = 0.1
    elif level_B == "All":
        alpha = 0.8
        radius_size = 0.08
    else:
        alpha = 0.9
        radius_size = 0.05

    left = update_colours(
        skill_hierarchy_df, left_gp, col_by=col_by, radius_size=radius_size, alpha=alpha
    )

    left.data_source.data = filter_skills(skill_hierarchy_df, level_A, level_B, level_C)
    push_notebook()


# %%
source = ColumnDataSource(data=skill_hierarchy_df)

hover = HoverTool(
    tooltips=[
        ("Skill name", "@{Skill name}(@{Skill number})"),
        (
            "Number of sentences that created skill",
            "@{Number of sentences that created skill}",
        ),
        ("Hierarchy level A", "@{Hierarchy level A name} (@{Hierarchy level A})"),
        ("Hierarchy level B", "@{Hierarchy level B name} (@{Hierarchy level B})"),
        ("Hierarchy level C", "@{Hierarchy level C name} (@{Hierarchy level C})"),
    ]
)

left_gp = figure(
    plot_width=500,
    plot_height=500,
    tools=[hover, WheelZoomTool(), BoxZoomTool(), SaveTool()],
    title=f"Skills coloured by hierarchy level",
    toolbar_location="below",
)
left = update_colours(
    skill_hierarchy_df, left_gp, col_by="A", radius_size=0.1, alpha=0.5
)

# %%

# %%
# streamlit code
title = '<span style="color:blue; font-family:Courier New; text-align:center; font-size:40px;">Explore the Skills Taxonomy.</span>'
st.markdown(title, unsafe_allow_html=True)

abstract = '<span style="color:black; font-family:Courier New; text-align:center; font-size:20px;">This interactive graph is based on a new data-driven approach to building a UK skills taxonomy, improving upon the original approach developed in [Djumalieva and Sleeman (2018)](https://www.escoe.ac.uk/the-first-publicly-available-data-driven-skills-taxonomy-for-the-uk/). The new method improves on the original method as it does not rely on a predetermined list of skills, and can instead automatically detect previously unseen skills in online job adverts. These ‘skill sentences’ are then grouped to define distinct skills, and a hierarchy is formed. The resulting taxonomy contains 18,893 separate skills.</span>'
st.markdown(abstract, unsafe_allow_html=True)

instruction2 = '<span style="color:black; font-family:Courier New; font-style: italic; text-align:center; font-size:20px;">Interrogate the interactive skills hierarchy by choosing each skill level in the dropdown boxes. Hover over each point to investigate skills information:</span>'
st.markdown(instruction2, unsafe_allow_html=True)

hierarchy_levels = st.selectbox("Skill Granularity", ("A", "B", "C"))
level_a = st.selectbox("Level A Skills", ["All"] + sorted(skill_hierarchy_df["Hierarchy level A"].unique().tolist()))
level_b = st.selectbox("Level B Skills", ["All"] + sorted(skill_hierarchy_df["Hierarchy level B"].unique().tolist()))
level_c = st.selectbox("Level C Skills", ["All"] + sorted(skill_hierarchy_df["Hierarchy level C"].unique().tolist()))

p2 = gridplot([[left_gp]])
update(hierarchy_levels, level_a, level_b, level_c, radius_size=0.1)
st.bokeh_chart(p2)

links = '<span style="color:black; font-family:Courier New; text-align:center; font-size:15px;">Read the full technical report [here](https://docs.google.com/document/d/1ZHE6Z6evxyUsSiojdNatSa_yMDJ8_UlB1K4YD1AhGG8/edit) and the extended article [here](https://docs.google.com/document/d/14lY7niHD0lyYpBj8TtlFGMSA2q4p4u5hl6LnXE4HYLs/edit).</span>'
st.markdown(links, unsafe_allow_html=True)
