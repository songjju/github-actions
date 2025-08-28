from sqlalchemy import create_engine, Column, Integer, String, DateTime, Float
from sqlalchemy.ext.declarative import declarative_base
from sqlalchemy.orm import sessionmaker
from datetime import datetime

Base = declarative_base()

class DrawHistory(Base):
    """추첨 기록 테이블"""
    __tablename__ = 'draw_history'
    
    id = Column(Integer, primary_key=True)
    draw_no = Column(Integer, unique=True)
    draw_date = Column(DateTime)
    jo = Column(Integer)
    number = Column(String(6))
    bonus_number = Column(String(6))
    created_at = Column(DateTime, default=datetime.now)

class PredictionHistory(Base):
    """예측 기록 테이블"""
    __tablename__ = 'prediction_history'
    
    id = Column(Integer, primary_key=True)
    draw_no = Column(Integer)  # 예측한 회차
    predicted_jo = Column(Integer)
    predicted_number = Column(String(6))
    confidence_score = Column(Float)
    prediction_type = Column(String(50))  # ML, Pattern 등
    created_at = Column(DateTime, default=datetime.now)
    
    # 나중에 실제 결과와 비교
    actual_jo = Column(Integer, nullable=True)
    actual_number = Column(String(6), nullable=True)
    is_correct = Column(Integer, nullable=True)  # 맞춘 자리수

class DatabaseManager:
    def __init__(self, db_path='lotto_predictions.db'):
        self.engine = create_engine(f'sqlite:///{db_path}')
        Base.metadata.create_all(self.engine)
        Session = sessionmaker(bind=self.engine)
        self.session = Session()
    
    def save_draw(self, draw_data):
        """추첨 데이터 저장"""
        draw = DrawHistory(
            draw_no=draw_data['draw_no'],
            draw_date=draw_data['draw_date'],
            jo=draw_data['jo'],
            number=draw_data['number'],
            bonus_number=draw_data.get('bonus_number', '')
        )
        self.session.add(draw)
        self.session.commit()
    
    def save_prediction(self, prediction_data):
        """예측 데이터 저장"""
        prediction = PredictionHistory(**prediction_data)
        self.session.add(prediction)
        self.session.commit()
        return prediction.id
    
    def get_prediction_history(self, limit=10):
        """예측 기록 조회"""
        return self.session.query(PredictionHistory)\
            .order_by(PredictionHistory.created_at.desc())\
            .limit(limit).all()
    
    def update_prediction_result(self, draw_no, actual_jo, actual_number):
        """예측 결과 업데이트"""
        predictions = self.session.query(PredictionHistory)\
            .filter_by(draw_no=draw_no).all()
        
        for pred in predictions:
            pred.actual_jo = actual_jo
            pred.actual_number = actual_number
            
            # 정확도 계산
            correct_digits = sum(1 for i, digit in enumerate(pred.predicted_number) 
                               if i < len(actual_number) and digit == actual_number[i])
            pred.is_correct = correct_digits
        
        self.session.commit()