# 🚀 2026 WorldCup Agent Backend

2026년 월드컵 채팅 봇 백엔드 입니다.

## 🔧 Configuration & Build

### 1. 인증서 생성

Git Bash에서 다음 명령어를 실행하여 SSL 인증서를 생성합니다:

```bash
# 인증서 디렉토리 생성
mkdir certs

# SSL 인증서 및 키 생성 (Git Bash - 슬래시 2개 필수)
openssl req -x509 -newkey rsa:2048 -nodes -keyout certs/key.pem -out certs/cert.pem -days 365 -subj "//CN=localhost"
```

### 2. 벡터 데이터베이스 및  SQL 다운로드

Chroma 벡터 데이터베이스를 다운로드하여 프로젝트 루트에 배치합니다:

**다운로드 링크:** https://drive.google.com/file/d/1ywS58DBRb6uRHOADY90_YhfZPie0H3lS/view?usp=drive_link

다운로드 후 압축을 해제하여 프로젝트 루트 폴더에 배치하세요.

---
PostgreSQL 등의 관계형 데이터베이스 초기 설정을 위해 아래 SQL 파일을 다운로드합니다:

**SQL 파일 다운로드 링크:**  
https://drive.google.com/file/d/1iPpchJEP-YvjEGctyG-bauBizR9zHkZ0/view?usp=sharing


---

다운로드한 `.sql` 파일을 원하는 데이터베이스에 import 하여 초기 데이터를 구성할 수 있습니다:

```bash
psql -U {your_rdb_user} -d {your_rdb_name} -f init_worldcup.sql
```
---

### 3. 환경 변수 설정

`.env` 파일을 생성하고 다음 내용을 작성합니다:

```env
# 🔑 Upstage LLM API 키
UPSTAGE_API_KEY={your_upstage_api_key}

# 🔍 Tavily 검색 API 키
TAVILY_API_KEY={your_tavily_api_key}

# 📦 Chroma Vector DB 설정
DB_NAME=spot_collection_2           # Chroma DB 이름
DB_PATH=./chroma_spot_v2            # 로컬 벡터 DB 저장 경로

# 🌐 Papago 번역 API 설정
CLIENT_ID={your_papago_client_id}           # Naver Papago 애플리케이션 Client ID
CLIENT_SECRET={your_papago_client_secret}   # Client Secret
TEXT_TRANSLATION_URL=https://papago.apigw.ntruss.com/nmt/v1/translation  # 번역 API URL

# 🗄️ RDB (예: PostgreSQL) 설정
RDB_HOST={your_rdb_host}             # 데이터베이스 호스트 주소
RDB_PORT={your_rdb_port}             # 데이터베이스 포트 (예: 5432)
RDB_NAME={your_rdb_name}             # 데이터베이스 이름
RDB_USER={your_rdb_user}             # DB 사용자 이름
RDB_PASSWORD={your_rdb_password}     # DB 비밀번호
```

### 4. 애플리케이션 실행

Docker Compose를 사용하여 애플리케이션을 빌드하고 실행합니다:

```bash
docker-compose up --build
```

### 5. 접속 확인

애플리케이션이 성공적으로 실행되면 다음 방법으로 접속할 수 있습니다:

**브라우저 접속:**
- URL: `https://localhost`
- 자체 서명 인증서이므로 브라우저 보안 경고가 나타날 수 있습니다. "고급" → "localhost로 이동"을 클릭하여 진행하세요.

**cURL 테스트:**
```bash
curl -k https://localhost
```

## 📁 프로젝트 구조

```
worldcup-2026-agent-backend/
├── certs/
│   ├── cert.pem              # SSL 인증서 (공개키)
│   └── key.pem               # SSL 개인키 (비밀키)
├── chroma_jinxes/            # 벡터 DB - jinxes
├── chroma_rules/             # 벡터 DB - rules
├── chroma_spot_v2/           # 벡터 DB - spot
├── .env                      # 환경 변수 설정 파일
├── main.py                   # FastAPI 애플리케이션
├── Dockerfile                # 도커 이미지 빌드 정의
├── docker-compose.yml        # 도커 컴포즈 설정
├── requirements.txt          # Python 패키지 목록
└── README.md                 # 프로젝트 문서
```