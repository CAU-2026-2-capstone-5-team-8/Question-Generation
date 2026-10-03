# 문제 본문 표시 형식 — markdown-latex-v1

문제는 기존 JSON의 `stem`, `choices`, `correct_choice_index`, `explanation`과 개념·목표·근거 정보를 유지한다.
텍스트 필드 안에서 문단, 강조, 목록과 LaTeX 수식을 사용한다. 문항 전체를 Markdown 파일로 대체하지 않는다.

## 생성 규칙

- 인라인 수식은 `\(...\)`, 별도 줄 수식은 `\[...\]`로 작성한다. 생성 단계에서는 달러 구분자를 쓰지 않는다.
- 실수 공간은 `\mathbb{R}^2`, 첨자는 `b_1`, 분수는 `\frac{1}{2}`, 곱셈은 `\cdot`로 작성한다.
- 행렬은 `\begin{bmatrix}1&2\\0&1\end{bmatrix}`로 작성한다. 긴 식은 별도 줄로 배치한다.
- JSON에서 역슬래시는 이스케이프한다. 예: `"stem": "In \\(\\mathbb{R}^2\\), choose a basis."`.
- HTML, 외부 이미지·링크, 코드 펜스는 생성하지 않는다. 구분자·중괄호·환경을 모두 닫는다.
- 수식 표시를 바꾸면서 값·오답·정답 인덱스·측정 목표를 바꾸지 않는다.

`concept-question-generation-prompt-v2`가 이 규칙을 지시하고, Python의 가벼운 검사는 열린 구분자,
중괄호·환경 불일치, 잘못된 JSON 제어문자, 명백한 ASCII 수식 표기를 거른다. 실제 LaTeX 명령의 지원 여부는
프론트와 같은 KaTeX 렌더러로 확인한다. 구조 검사는 수학적 정답 검토를 대신하지 않는다.

## 생성 → 표시 검사 → 사람 검토

```sh
bookmatch-question-generation concept local-drafts \
  --blueprint ../ML/data/output/concept-assessment-v2/blueprint.json \
  --drafts examples/concept-local-drafts.json \
  --output-dir data/generated/concept-assessment-math-v2

npm --prefix ../Frontend/native run validate:questions -- \
  ../../Question-Generation/data/generated/concept-assessment-math-v2
```

두 번째 명령의 경로는 `Frontend/native` 기준이다. 렌더링 오류가 있으면 비정상 종료한다.
Gemini 생성 결과에도 같은 검사를 적용한다. API 없는 로컬 초안은 계속 `codex-local-draft`로 기록한다.

기존 v1 산출물은 그대로 읽을 수 있다. 새 생성 명령은 이전 프롬프트 산출물을 현재 프롬프트의 결과로 재사용하지
않고 새 출력 폴더를 요구한다. 수정된 텍스트는 새 content ID를 받으며 이전 승인·발급 스냅샷을 덮어쓰지 않는다.

## 프론트

공통 `QuestionContent`를 진단 화면과 문항 검토의 질문·보기·해설·측정 목표에 적용한다.
웹은 KaTeX의 HTML+MathML을 표시하고 RN iOS/Android는 같은 결과를 내부 WebView에 표시한다.
KaTeX 폰트와 스타일은 앱에 포함하며 수식을 외부 서버에 보내지 않는다. 일반 텍스트는 그대로 표시한다.
잘못된 수식은 원문과 확인 안내를 남기고 다른 문제 조작을 유지한다.

수식 표시 검사는 내용 승인과 별도다. 2026-10-04에는 사용자의 위임에 따라 새 v4 후보 18개를
AI가 내용 검토했으며 `ai-question-review-v1`로 기록했다. 실제 판정은
[AI 검토 기록](concept-ai-review-2026-10-04.md)을 참고한다. AI 판정을 사람 승인으로 표시하지 않는다.
