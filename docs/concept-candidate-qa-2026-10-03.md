# 선형대수 후보 18문항의 사전 검토

검토자는 AI(Codex)이며 **사람 승인 기록이 아니다**. 실제 bank는 기존 문항을 유지한다.
검토 기준은 `concept-assessment-math-v2/review.md`와 각 생성 JSON의 질문·보기·해설·측정 목표다.
근거 목차는 개념 선정용이며 정답의 수학적 근거나 본문 난이도로 사용하지 않았다.

## 발견과 수정

basis / meaning의 “Which two conditions must …”는 필요조건만 고르라는 뜻으로도 읽힐 수 있다.
“Which pair of conditions is necessary and sufficient …”로 바꿔 기저를 정의하는 두 조건을 묻도록 했다.
보기·정답 인덱스·해설·개념 ID·측정 목표는 바꾸지 않았다. 원본 18개는 보존하고
`data/generated/concept-assessment-reviewed-draft-v3/`에 새 후보와 검토 자료를 생성했다.
두 버전 비교 결과 basis / meaning의 `stem`, `generated_question_id`만 변경됐다.
폴더 이름의 reviewed-draft는 이 AI 사전 검토를 의미하며 human-approved를 뜻하지 않는다.

재현(Question-Generation 루트):

```sh
.venv/bin/bookmatch-question-generation concept local-drafts \
  --blueprint ../ML/data/output/concept-assessment-v2/blueprint.json \
  --drafts examples/concept-local-drafts.json \
  --output-dir data/generated/concept-assessment-reviewed-draft-v3
```

설계서 SHA-256: `93c8995a232f7a6a962522b1bb26ebda4923d2121f20e5b047fead2967d3b5c2`.
설계서는 canonical 목차에서 생성한 ignored 산출물이다. 원본 책 텍스트나 실제 승인 기록을 커밋하지 않는다.

## 내용 확인

다음은 해설과 계산을 재검토한 결과다. 정답 번호는 1부터 센다. 타당성·변별력의 실증 검사가 아니다.

| 개념 | 의미 | 적용 | 추론 |
| --- | --- | --- | --- |
| matrix | ② 행 수 × 열 수 | ③ 첫 원소는 1×2+2×3=8 | ④ AB=diag(1,0), BA=diag(0,1) |
| vector | ① 대응 성분 일치 | ③ 2u−3v=(5,1) | ② w=2u+3v |
| linear system | ④ 중복 제약, 무한해 | ① (3,2)가 두 식을 만족 | ③ 마지막 행은 0=1 |
| linear independence | ② 영조합의 계수가 모두 0 | ④ 2u−v=0, 비자명 계수 | ① 세 벡터의 비자명 관계 |
| basis | ③ 생성 + 독립; 질문 표현 수정 | ② 좌표 (3,1)로 (4,2) 복원 | ④ 생성은 유지, 독립은 깨짐 |
| orthogonality | ① 내적 0 | ③ k−2=0 → k=2 | ② 자기 내적의 양수성으로 각 계수 0 |

18개 모두 prior-knowledge이며 선택형 추론은 올바른 논증을 알아보는 수행이다. 직접 증명을 작성하는 능력과 동일시하지 않는다.
정답 위치 분포는 ①4 / ②5 / ③5 / ④4다. 위치 빈도가 비슷해도 검토 문서 순서의 반복 패턴과 보기 길이로
정답을 추측할 가능성은 별도로 사람 검토가 필요하다. 자동 검사 통과를 높은 문항 품질로 해석하지 않는다.

## 검사와 사람 검토의 남은 범위

- 생성기 127개 검사 통과, 외부 API live 2개 제외, Ruff/형식 검사 통과.
- 새 후보 18개·텍스트 108개 필드의 실제 KaTeX 표시 검사 통과(수식 오류 0).
- 로컬 후보만 재생성했으며 Gemini 호출은 하지 않았다.
- 수학 전문가/사용자의 단일 정답·오답 타당성, 의미/적용/추론 구분, 읽기 부담, 보기 길이 단서 검토가 남아 있다.
- 영어 원문으로 인한 언어 부담을 수학 사전 지식 부족으로 해석하지 않아야 한다. 한국어 표현 검토는 별도 버전 작업이다.
- 설계 난도 1/2는 실제 응답으로 보정한 난도가 아니다. 학생 응답, 변별도, 신뢰도와 추천 효과는 아직 측정하지 않았다.

사람 검토 결과를 실제 `HumanQuestionReview`로 받은 후 승인된 생성 ID만 import한다. 사전 검토 결과를
`approve`로 바꾸거나 검토자 이름·시간을 만들어 넣지 않는다. 이전 발급 snapshot과 사용자 응답은 보존한다.
