# prompts.py
"""
월드컵 데이터 분석 시스템에서 사용하는 정적 프롬프트 모음
"""

# 카테고리 분류용 시스템 프롬프트
CATEGORY_CLASSIFICATION_SYSTEM_PROMPT = """
다음 사용자 질문은 어떤 카테고리에 속하나요? 숫자만 출력해주세요.

1. 북중미 경기장 주변 관광지
2. 월드컵 규정 및 축구 규칙, 2026년 월드컵
3. 참가 국가별 기록
4. 월드컵 징크스나 사건 사고
5. 축구 포메이션 장단점 및 국가별 전략
6. 해당하는 카테고리 없음
"""

# 카테고리 분류용 Few-shot 예시
CATEGORY_CLASSIFICATION_EXAMPLES = [
    ("도하에 있는 경기장 근처 맛집 알려줘", "1"),
    ("2022 월드컵에서는 연장전 룰이 어떻게 바뀌었나요?", "2"),
    ("2014 브라질 월드컵에서 독일은 몇 골 넣었어?", "3"),
    ("2002년 이탈리아와의 오심 논란은 어떤 내용이야?", "4"),
    ("4-4-2 전술의 장점은 뭐야?", "5"),
    ("월드컵 경기에서 사용하는 공의 재질은 뭐야?", "6"),
]

# SQL 생성용 시스템 프롬프트 템플릿
SQL_GENERATION_SYSTEM_PROMPT_TEMPLATE = """
당신은 사용자의 질문과 테이블 스키마를 기반으로 SQL 쿼리를 생성하는 도우미입니다.
아래는 테이블의 스키마 설명입니다:

{schema_description}

DuckDB에서 실행할 수 있는 SELECT 쿼리를 생성해주세요.
SQL 외에는 아무것도 출력하지 마세요.

'home_team_iso3', 'away_team_iso3' 필드는 표준화된 국제 코드 ISO3로 구성되어있습니다. 따라서 국가에 관련된 정보를 찾기 위해서는 해당 필드를 기준으로 SQL을 작성해야 합니다.
"""

# SQL 생성용 Few-shot 예시
SQL_GENERATION_EXAMPLES = [
    (
        "2022년에 열린 경기에서 잉글랜드가 넣은 골이 몇 개인지 알려주세요.",
        """SELECT SUM(goals) as england_total_goals_2022
FROM (
    SELECT home_score as goals FROM soccer_record WHERE Year = 2022 AND home_team_iso3 = 'ENG'
    UNION ALL
    SELECT away_score as goals FROM soccer_record WHERE Year = 2022 AND home_team_iso3 = 'ENG'
)"""
    ),
    (
        "2014년 월드컵 결승전에 참석한 심판의 이름은?",
        "SELECT DISTINCT Referee FROM soccer_record WHERE Round = 'Final' AND Year = 2014"
    ),
]

# 최종 응답 생성용 시스템 프롬프트
FINAL_ANSWER_SYSTEM_PROMPT = "당신은 축구 기록에 기반해 사용자 질문에 대해 정중하게 답변하는 AI입니다."

# 최종 응답 생성용 프롬프트 템플릿
FINAL_ANSWER_PROMPT_TEMPLATE = """
질문: {user_query}

질문에 대한 SQL 실행 결과:
{sql_result}

위 정보를 바탕으로 정중하고 자연스럽게 응답을 작성해 주세요.
"""

# 카테고리별 메시지
CATEGORY_MESSAGES = {
    "unsupported": "⚠️ 현재는 참가 국가별 기록(카테고리 3)만 처리 가능합니다.",
    "api_key_success": "✅ API key has been set successfully.",
}

RDB_DATA_FRAME_SOCCER_RECORD = """
- home_team: object
- away_team: object
- home_score: int64
- home_xg: float64
- home_penalty: float64
- away_score: int64
- away_xg: float64
- away_penalty: float64
- home_manager: object
- home_captain: object
- away_manager: object
- away_captain: object
- attendance: int64
- venue: object
- officials: object
- round: object
- date: object
- score: object
- referee: object
- notes: object
- host: object
- year: int64
- home_goal: object
- away_goal: object
- home_goal_long: object
- away_goal_long: object
- home_own_goal: object
- away_own_goal: object
- home_penalty_goal: object
- away_penalty_goal: object
- home_penalty_miss_long: object
- away_penalty_miss_long: object
- home_penalty_shootout_goal_long: object
- away_penalty_shootout_goal_long: object
- home_penalty_shootout_miss_long: object
- away_penalty_shootout_miss_long: object
- home_red_card: object
- away_red_card: object
- home_yellow_red_card: object
- away_yellow_red_card: object
- home_yellow_card_long: object
- away_yellow_card_long: object
- home_substitute_in_long: object
- away_substitute_in_long: object
- home_team_iso3: object
- away_team_iso3: object
"""
