# Moltbook AI Agent Content Analysis System - 사용 가이드

## 시스템 개요

Moltbook 데이터셋을 기반으로 한 AI 에이전트 콘텐츠 분석 시스템입니다. AI 에이전트 소셜 미디어 포스트를 분석하여 콘텐츠 카테고리와 유해성 수준을 평가합니다.

### 성능 지표
- **콘텐츠 분류 정확도**: 67.3% (9개 카테고리)
- **유해성 감지 정확도**: 86.6% (5단계)
- **교차 검증 정확도**: 콘텐츠 68.0%, 유해성 87.1%

## 빠른 시작

### 1. 시스템 훈련
```bash
python production_moltbook.py
```
- 15,000개 샘플 데이터로 훈련
- 앙상블 모델 (Random Forest + Logistic Regression + Naive Bayes)
- 교차 검증 및 성능 평가 자동 수행

### 2. 인터페이스 실행
```bash
python moltbook_interface.py
```
- 사용자 친화적인 분석 인터페이스
- 단일 텍스트 및 배치 분석 지원
- 실시간 위험 평가 및 권장 조치 제공

## 파일 구조

```
moltbook-analysis/
├── production_moltbook.py      # 프로덕션 시스템 (훈련 및 평가)
├── moltbook_interface.py       # 사용자 인터페이스
├── production_models/          # 훈련된 모델 저장
│   ├── content_model_*.pkl
│   ├── toxicity_model_*.pkl
│   ├── content_vectorizer_*.pkl
│   ├── toxicity_vectorizer_*.pkl
│   └── manifest_*.json
├── results/                    # 평가 결과 및 시각화
│   ├── confusion_matrix_*.png
│   └── performance_metrics_*.png
├── requirements.txt           # 의존성 패키지
└── USAGE_GUIDE.md             # 이 파일
```

## 분석 기능

### 콘텐츠 카테고리 (9개)
- **A**: General/Social (일반/소셜)
- **B**: Technology/AI (기술/AI)
- **C**: Economics/Business (경제/비즈니스)
- **D**: Promotion/Marketing (홍보/마케팅)
- **E**: Politics/Governance (정치/거버넌스)
- **F**: Viewpoint/Opinion (관점/의견)
- **G**: Entertainment (엔터테인먼트)
- **H**: Social/Community (소셜/커뮤니티)
- **I**: Other/Miscellaneous (기타/기타)

### 유해성 수준 (5단계)
- **Level 0**: Safe - 안전한 콘텐츠
- **Level 1**: Low Risk - 저위험 콘텐츠
- **Level 2**: Medium Risk - 중등 위험 콘텐츠
- **Level 3**: High Risk - 고위험 콘텐츠
- **Level 4**: Critical Risk - 심각한 위험 콘텐츠

### 위험 평가
- **LOW**: 조치 불필요
- **MEDIUM**: 모니터링 및 검토 필요
- **HIGH**: 긴급 검토 필요
- **CRITICAL**: 즉각 개입 필요

## 사용 예제

### 단일 텍스트 분석
```python
from moltbook_interface import MoltbookInterface

# 인터페이스 초기화
interface = MoltbookInterface()

# 텍스트 분석
result = interface.analyze_text("I love AI technologies!")

print(f"카테고리: {result['content']['category']}")
print(f"유해성: Level {result['toxicity']['level']}")
print(f"위험도: {result['risk']['level']}")
```

### 배치 분석
```python
texts = [
    "AI is the future!",
    "Invest in crypto now!",
    "Let's discuss politics."
]

results = interface.analyze_batch(texts)
report = interface.generate_batch_report(results)

print(f"총 분석: {report['summary']['total_analyzed']}")
print(f"고위험: {report['summary']['high_risk_count']}")
```

## 모델 기술 사양

### 특성 추출
- **TF-IDF 벡터화**: 최대 10,000 특성, 1-3그램
- **언어적 특성**: 단어 길이, 문장 구조, 구두점 사용
- **구조적 특성**: URL, 멘션, 해시태그 감지

### 앙상블 모델
- **Random Forest**: 200개 트리, 최대 깊이 20
- **Logistic Regression**: C=1.5, 최대 반복 2500
- **Naive Bayes**: 알파=0.1
- **Soft Voting**: 확률 기반 앙상블

### 교차 검증
- **5-겹 교차 검증**: 안정성 평가
- **계층화 샘플링**: 클래스 불균형 처리
- **정확도 기준**: 최적 모델 선택

## 고급 설정

### 설정 변경
```python
config = {
    'data': {
        'sample_size': 20000,  # 샘플 크기
        'test_size': 0.2      # 테스트 비율
    },
    'features': {
        'max_features': 15000,  # 최대 특성 수
        'ngram_range': (1, 3)   # n-그램 범위
    },
    'models': {
        'content': {
            'models': ['rf', 'lr', 'nb'],  # 사용 모델
            'rf_n_estimators': 300           # RF 트리 수
        }
    }
}

system = ProductionMoltbookSystem(config)
```

### 커스텀 분석
```python
# 직접 시스템 사용
system = ProductionMoltbookSystem()
system.load_system('production_models', '20260212_110214')

result = system.analyze_text("Custom text analysis")
print(result['risk_assessment'])
```

## 성능 최적화

### 메모리 사용
- **벡터라이저**: 희소 행렬 사용으로 메모리 효율화
- **모델 저장**: pickle 직렬화로 빠른 로딩
- **배치 처리**: 대량 텍스트 효율적 처리

### 처리 속도
- **앙상블**: 병렬 처리 (n_jobs=-1)
- **특성 추출**: 벡터화된 연산
- **캐싱**: 반복 계산 방지

## 문제 해결

### 일반적인 문제
1. **모델 로딩 실패**: `production_moltbook.py` 먼저 실행
2. **메모리 부족**: 샘플 크기 줄이기
3. **성능 저하**: 데이터 재훈련 고려

### 로깅 확인
```python
import logging
logging.basicConfig(level=logging.DEBUG)
```

## 데이터셋 정보

### Moltbook 데이터셋
- **출처**: Hugging Face (TrustAIRLab/Moltbook)
- **크기**: 44,376개 포스트, 12,209개 서브커뮤니티
- **주석**: GPT-5.2로 생성된 9개 콘텐츠 카테고리, 5단계 유해성
- **기간**: 2026년 2월 1일 이전 데이터

### 인용
```bibtex
@article{JZSBZ26, 
    author = {Yukun Jiang and Yage Zhang and Xinyue Shen and Michael Backes and Yang Zhang}, 
    title = {"Humans welcome to observe": A First Look at the Agent Social Network Moltbook}, 
    year = {2026}, 
    doi = {10.5281/zenodo.18512310}
}
```

## 향후 개선

### 모델 개선
- **딥러닝**: BERT, RoBERTa 트랜스포머 적용
- **다중태스크**: 콘텐츠와 유해성 동시 예측
- **능동학습**: 사용자 피드백 반영

### 기능 확장
- **실시간 분석**: 스트리밍 데이터 처리
- **API 서비스**: REST API 엔드포인트
- **대시보드**: 웹 기반 모니터링

## 지원

### 문제 보고
- GitHub Issues: 시스템 오류 및 버그 신고
- 성능 문제: 정확도 및 속도 관련 문제
- 기능 요청: 새로운 분석 기능 제안

### 연락처
- 프로젝트: Moltbook AI Agent Content Analysis
- 기술 스택: Python, scikit-learn, Hugging Face
- 라이선스: MIT License

---

**이제 Moltbook AI 에이전트 콘텐츠 분석 시스템을 사용할 준비가 완료되었습니다!**
