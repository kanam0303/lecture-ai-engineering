import streamlit as st
import yfinance as yf
import pandas as pd
import numpy as np
from datetime import datetime, timedelta
import plotly.graph_objects as go

# アプリのタイトル
st.title('株価チャート表示')

# サイドバーで株式シンボルを入力
stock_symbol = st.sidebar.text_input('銘柄（例：^GSPC, AAPL）:', '^GSPC')

# 期間の選択
period_options = ['1週間', '1ヶ月', '3ヶ月', '6ヶ月', '1年', '5年']
selected_period = st.sidebar.selectbox('期間を選択:', period_options)

# 期間の文字列を日数に変換
period_days = {
    '1週間': 7,
    '1ヶ月': 30,
    '3ヶ月': 90,
    '6ヶ月': 180,
    '1年': 365,
    '5年': 365 * 5
}

# 日付範囲の計算
end_date = datetime.now()
start_date = end_date - timedelta(days=period_days[selected_period])

# データの取得
@st.cache_data
def load_data(symbol, start, end):
    try:
        data = yf.download(symbol, start=start, end=end)
        # 欠損値があれば前の値で埋める
        data = data.fillna(method='ffill')
        return data
    except Exception as e:
        st.error(f"データの取得に失敗しました: {e}")
        return None

# メインコンテンツ
if stock_symbol:
    data = load_data(stock_symbol, start_date, end_date)
    
    if data is not None and not data.empty:
        # 銘柄情報の取得
  
        ticker = yf.Ticker(stock_symbol)
        info = ticker.info
        company_name = info.get('longName', stock_symbol)
        st.subheader(f"{company_name} ({stock_symbol})")

        # Streamlitネイティブのチャートを表示
        st.subheader("株価推移")
        st.line_chart(data['Close'])
        
        # 出来高チャートも表示
        if 'Volume' in data.columns:
            st.subheader("出来高")
            st.bar_chart(data['Volume'])
        
        # 株価データの表示
        st.subheader("株価データ")
        st.dataframe(data)
        
    else:
        st.warning(f"{stock_symbol}のデータを取得できませんでした。正しい銘柄を入力してください。")
else:
    st.info("左サイドバーに銘柄を入力してください。")

# アプリの説明
st.sidebar.markdown("""
### 使用方法
1. 左のサイドバーに銘柄を入力
2. 表示したい期間を選択
3. チャートと株価データが表示されます
""")

# フッター
st.markdown("---")
st.caption("データ提供: Yahoo Finance (yfinance)")
st.caption("※ 投資は自己責任で行ってください。")
