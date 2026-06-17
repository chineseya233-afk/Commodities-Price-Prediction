"""
Database models and connection management (Backend Agent)

Uses SQLite for POC simplicity with SQLAlchemy ORM.
"""

from sqlalchemy import (
    Column, Integer, Float, String, DateTime, Boolean, Text, JSON,
    create_engine,
)
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()


class PriceData(Base):
    """Historical commodity price records."""
    __tablename__ = "price_data"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    date = Column(String, index=True, nullable=False)
    commodity = Column(String, index=True, nullable=False, default="diesel_0")
    price = Column(Float, nullable=False)
    open_price = Column(Float)
    high = Column(Float)
    low = Column(Float)
    volume = Column(Integer)
    source = Column(String, default="simulator")
    is_outlier = Column(Boolean, default=False)
    created_at = Column(DateTime, default=datetime.utcnow)


class PredictionResult(Base):
    """Model prediction outputs."""
    __tablename__ = "prediction_results"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    prediction_date = Column(String, nullable=False)  # when prediction was made
    target_date = Column(String, nullable=False)  # date being predicted
    commodity = Column(String, default="diesel_0")
    model_name = Column(String, nullable=False)  # tft, prophet, xgboost, naive
    predicted_p10 = Column(Float)
    predicted_p50 = Column(Float)
    predicted_p90 = Column(Float)
    actual_price = Column(Float)  # filled later when actual data arrives
    mape = Column(Float)
    is_qa_passed = Column(Boolean, default=True)
    qa_notes = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class AnalysisReport(Base):
    """LLM-generated analysis reports."""
    __tablename__ = "analysis_reports"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    report_date = Column(String, nullable=False)
    commodity = Column(String, default="diesel_0")
    report_type = Column(String, default="daily")  # daily, weekly
    summary = Column(Text)
    trend_analysis = Column(Text)
    risk_factors = Column(JSON)
    procurement_advice = Column(JSON)
    data_quality_notes = Column(Text)
    raw_llm_response = Column(Text)
    created_at = Column(DateTime, default=datetime.utcnow)


class User(Base):
    """System users with RBAC."""
    __tablename__ = "users"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    username = Column(String, unique=True, nullable=False)
    hashed_password = Column(String, nullable=False)
    role = Column(String, default="procurement")  # admin, executive, procurement
    full_name = Column(String)
    is_active = Column(Boolean, default=True)
    created_at = Column(DateTime, default=datetime.utcnow)


class ModelMetrics(Base):
    """Historical model performance tracking."""
    __tablename__ = "model_metrics"
    
    id = Column(Integer, primary_key=True, autoincrement=True)
    evaluation_date = Column(String, nullable=False)
    model_name = Column(String, nullable=False)
    mape = Column(Float)
    rmse = Column(Float)
    mae = Column(Float)
    directional_accuracy = Column(Float)
    coverage_rate = Column(Float)
    created_at = Column(DateTime, default=datetime.utcnow)


# 数据库引擎和会话设置
def get_engine(database_url: str = "sqlite:///./data/commodity_prediction.db"):
    """Create database engine."""
    return create_engine(database_url, echo=False)


def create_tables(engine):
    """Create all database tables."""
    Base.metadata.create_all(engine)


def get_session(engine):
    """Create a database session."""
    Session = sessionmaker(bind=engine)
    return Session()
