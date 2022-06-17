import streamlit as st
import requests
import numpy as np
import pandas as pd


if st.button('网络各层logα分布示意图'):
    col1, col2, col3 = st.columns(3)
    with col1:
        st.image("./vgg_figs/0.png")

    with col2:
        st.image("./vgg_figs/1.png")

    with col3:
        st.image("./vgg_figs/2.png")

    col4, col5, col6 = st.columns(3)
    with col4:
        st.image("./vgg_figs/3.png")

    with col5:
        st.image("./vgg_figs/4.png")

    with col6:
        st.image("./vgg_figs/5.png")

    col7, col8, col9 = st.columns(3)
    with col7:
        st.image("./vgg_figs/6.png")

    with col8:
        st.image("./vgg_figs/7.png")

    with col9:
        st.image("./vgg_figs/8.png")

    col10, col11, col12 = st.columns(3)
    with col10:
        st.image("./vgg_figs/9.png")

    with col11:
        st.image("./vgg_figs/10.png")

    with col12:
        st.image("./vgg_figs/11.png")

    col13, col14, col15 = st.columns(3)
    with col13:
        st.image("./vgg_figs/12.png")

    with col14:
        st.image("./vgg_figs/13.png")

    with col15:
        st.image("./vgg_figs/14.png")

    col16, col17, col18 = st.columns(3)
    with col16:
        st.image("./vgg_figs/15.png")

    with col17:
        st.image("./vgg_figs/16.png")

if st.button('模型压缩结果'):
    url = 'http://localhost:8000/vgg/'
    response = requests.get(url)
    data_received = response.json()
    # response.raise_for_status()  # raises exception when not a 2xx response
    # if response.status_code != 204:
    #     data_received = response.json()
    prune = data_received["prune"]
    pruneandbit = data_received["pruneandbit"]
    acc = data_received["acc"]
    st.write(f'剪枝后压缩率: {prune}')
    st.write(f'剪枝加量化后压缩率: {pruneandbit}')
    st.write(f'准确率: {acc}')