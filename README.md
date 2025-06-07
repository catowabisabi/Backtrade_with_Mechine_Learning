# Backtrade with Machine Learning
# 機器學習回測系統

![Build Status](https://img.shields.io/badge/build-passing-brightgreen)
![License](https://img.shields.io/badge/license-MIT-blue)
![Stars](https://img.shields.io/github/stars/yourusername/your-repo?style=social)

[English](#english) | [繁體中文](#繁體中文)

---

<a name="english"></a>
# English Documentation

## 📖 Introduction
A comprehensive backtesting system that combines traditional trading strategies with machine learning approaches. This project provides tools for developing, testing, and evaluating trading strategies using both technical analysis and AI/ML models.

## 🎯 Purpose
- Implement and test various trading strategies
- Integrate machine learning models for market prediction
- Evaluate strategy performance through backtesting
- Provide visualization tools for analysis

## ✨ Features
- 🤖 Machine Learning Integration
- 📊 Technical Analysis
- 🔄 Automated Backtesting
- 📈 Performance Visualization
- 🛠️ Strategy Development Tools

## 📂 Project Structure
```
project/
├── notebooks/                # Jupyter notebooks and Python scripts
│   ├── backtest_01.py       # Basic backtesting implementation
│   ├── backtest_03.py       # Enhanced backtesting with Golden Cross
│   ├── backtest_04.py       # Advanced strategy testing
│   ├── backtest_04.ipynb    # Interactive strategy analysis
│   ├── backtest_05_AI.ipynb # AI model integration
│   └── backtest_05B_AI.ipynb# Enhanced AI backtesting
├── functional_class/        # Core functionality classes
├── services/               # Trading services
├── utils/                 # Utility functions
└── tests/                # Test modules
```

## 📓 Notebooks Description

### 1. Basic Backtesting (backtest_01.py)
- Basic implementation of backtesting framework
- Stock data loading and processing
- Simple strategy testing capabilities
- Suitable for beginners to understand the basics

### 2. Golden Cross Strategy (backtest_03.py)
- Implementation of Golden Cross trading strategy
- Enhanced data processing
- Performance evaluation
- Technical indicator integration

### 3. Advanced Strategy Testing
#### Basic Version (backtest_04.py, backtest_04.ipynb)
- Complex strategy implementation
- Interactive analysis and visualization
- Performance metrics calculation
- Risk management integration

#### Extended Version (backtest_04_extended.ipynb)
- Enhanced performance metrics
- Portfolio optimization
- Advanced risk management
- Custom indicator development
- Multiple timeframe analysis

### 4. AI Integration
#### Basic AI Implementation (backtest_05_AI.ipynb)
- Machine learning model integration
- Feature engineering pipeline
- AI-based prediction system
- Model evaluation framework
- Basic AI strategy implementation

#### Advanced AI Features (backtest_05B_AI.ipynb)
- Multi-model ensemble methods
- Hyperparameter optimization
- Advanced feature selection
- Real-time prediction integration
- Enhanced strategy optimization
- Comprehensive risk management

## ⚙️ Installation
```bash
# Clone the repository
git clone https://github.com/yourusername/Backtrade_with_Mechine_Learning.git

# Create virtual environment
python -m venv venv
source venv/bin/activate  # Linux/Mac
.\venv\Scripts\activate   # Windows

# Install dependencies
pip install -r requirements.txt
```

## 🚀 Usage
1. Start with basic backtesting notebooks
2. Progress to advanced strategy testing
3. Explore AI integration notebooks
4. Develop custom strategies

## 📄 License
This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

---

<a name="繁體中文"></a>
# 繁體中文文檔

## 📖 簡介
一個綜合性的回測系統，結合傳統交易策略和機器學習方法。本項目提供工具用於開發、測試和評估使用技術分析和人工智能/機器學習模型的交易策略。

## 🎯 目標
- 實現和測試各種交易策略
- 整合機器學習模型進行市場預測
- 通過回測評估策略表現
- 提供分析視覺化工具

## ✨ 功能特點
- 🤖 機器學習整合
- 📊 技術分析
- 🔄 自動化回測
- 📈 績效視覺化
- 🛠️ 策略開發工具

## 📂 專案結構
```
project/
├── notebooks/                # Jupyter 筆記本和 Python 腳本
│   ├── backtest_01.py       # 基礎回測實現
│   ├── backtest_03.py       # 黃金交叉策略強化版
│   ├── backtest_04.py       # 進階策略測試
│   ├── backtest_04.ipynb    # 互動式策略分析
│   ├── backtest_05_AI.ipynb # AI 模型整合
│   └── backtest_05B_AI.ipynb# 強化版 AI 回測
├── functional_class/        # 核心功能類
├── services/               # 交易服務
├── utils/                 # 工具函數
└── tests/                # 測試模組
```

## 📓 筆記本說明

### 1. 基礎回測 (backtest_01.py)
- 回測框架的基礎實現
- 股票數據加載和處理
- 簡單策略測試功能
- 適合初學者理解基礎概念

### 2. 黃金交叉策略 (backtest_03.py)
- 黃金交叉交易策略實現
- 強化數據處理
- 績效評估
- 技術指標整合

### 3. 進階策略測試
#### 基礎版本 (backtest_04.py, backtest_04.ipynb)
- 複雜策略實現
- 互動式分析和視覺化
- 績效指標計算
- 風險管理整合

#### 擴展版本 (backtest_04_extended.ipynb)
- 增強的績效指標
- 投資組合優化
- 進階風險管理
- 自定義指標開發
- 多時間框架分析

### 4. AI 整合
#### 基礎 AI 實現 (backtest_05_AI.ipynb)
- 機器學習模型整合
- 特徵工程流程
- AI 預測系統
- 模型評估框架
- 基礎 AI 策略實現

#### 進階 AI 功能 (backtest_05B_AI.ipynb)
- 多模型集成方法
- 超參數優化
- 進階特徵選擇
- 實時預測整合
- 增強策略優化
- 綜合風險管理

## ⚙️ 安裝方式
```