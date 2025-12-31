import streamlit as st
import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
import pickle
import io

st.set_page_config(page_title="配置馬券 AI学習センター", layout="wide")

st.title("🧠 配置馬券 AI学習・モデル作成")
st.write("集めた100R分のCSVをここに全て投入して、AIモデル（model.pkl）を作成します。")

# 1. ファイルアップローダー（複数選択可能）
uploaded_files = st.file_uploader("着順入りのCSVファイルを全て選択してください", type='csv', accept_multiple_files=True)

if uploaded_files:
    dfs = []
    for file in uploaded_files:
        df = pd.read_csv(file)
        dfs.append(df)
    
    # 全データを1つに統合
    all_df = pd.concat(dfs, ignore_index=True)
    st.success(f"合計 {len(uploaded_files)} 個のファイルを統合し、{len(all_df)} 行のデータを読み込みました。")

    # データの中身を少し確認
    st.subheader("📊 統合データのプレビュー")
    st.dataframe(all_df.head())

    # 2. 学習の準備（前処理）
    st.divider()
    st.subheader("⚙️ 学習設定")

    # 学習に使う項目（特徴量）の選択
    # 配置術において重要な項目を数値化します
    features = ['正番', '単ｵｯｽﾞ', '総合スコア'] # スコアやオッズは必須
    
    # 文字列の「属性」や「パターン」をAIが扱えるように簡易変換（今回はスコアに集約されている前提）
    # もし特定のパターンの有無を学習させたい場合はここで加工します。

    if st.button("🚀 AI学習を開始する"):
        try:
            # --- 学習用データの作成 ---
            # 着順を「3着以内(1)」か「それ以外(0)」に変換
            all_df['target'] = all_df['着順'].apply(lambda x: 1 if x <= 3 else 0)
            
            # 不要なデータ（欠損値）を削除
            train_df = all_df.dropna(subset=features + ['target'])
            
            X = train_df[features] # ヒント
            y = train_df['target'] # 正解
            
            # --- AI学習実行 (ランダムフォレスト) ---
            model = RandomForestClassifier(n_estimators=100, random_state=42)
            model.fit(X, y)
            
            st.success("✅ 学習が完了しました！")

            # 3. モデルの保存（model.pkl）
            # メモリ上に保存してダウンロード可能にする
            model_byte = io.BytesIO()
            pickle.dump(model, model_byte)
            
            st.download_button(
                label="📥 完成したモデル (model.pkl) をダウンロード",
                data=model_byte.getvalue(),
                file_name="model.pkl",
                mime="application/octet-stream"
            )
            
            # AIの重要度診断
            st.subheader("💡 AIが重視した項目")
            importance = pd.DataFrame({'項目': features, '重要度': model.feature_importances_})
            st.bar_chart(importance.set_index('項目'))

        except Exception as e:
            st.error(f"学習中にエラーが発生しました: {e}")
            st.info("ヒント: オッズやスコアに数字以外の文字が入っていないか確認してください。")

else:
    st.info("左上のボタンから、これまで集めたCSVファイルを全てアップロードしてください。")
