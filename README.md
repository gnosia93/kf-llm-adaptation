# 워크숍: 3B LLM 에 쿠버네티스 지식 넣기 — 계속 사전학습(CPT) 실습

> 공개된 쿠버네티스 한글 문서를 수집·정제해서, 사전학습된 3B 언어모델(Qwen2.5-3B)에
> **계속 사전학습(Continued Pre-Training)** 으로 도메인 지식을 넣는다.
> 학습 전후를 수치(perplexity)와 생성 결과로 비교한다.

## 이 워크숍에서 하는 것

```
 쿠버네티스 한글 문서 569개 (GitHub kubernetes/website)
          │  ① 수집 · 정제 (Hugo 문법 제거)
          ▼
 2,048 토큰 블록 922개 (약 114만 토큰) + 일반 한국어 위키 20%
          │  ② 계속 사전학습: full fine-tuning, L40S 48GB
          ▼
 k8s 도메인 적응 모델 ──③ 평가: perplexity 3종 + 이어쓰기 비교
          │
          ▼  ④ 다음 단계: SFT / Chat Vector / RAG / 서빙
```

## 대상과 사전 지식

| 필요 | 수준 |
|---|---|
| Python | 함수, 리스트/딕셔너리, 패키지 설치를 할 줄 앎 |
| 터미널 | `cd`, `ls`, `ssh` 정도 |
| 딥러닝 | "모델을 학습시킨다"는 개념 정도. 수식은 몰라도 됨 |
| 쿠버네티스 | 몰라도 됨. 다만 알면 결과를 해석하기가 더 재미있음 |
| AWS | 콘솔 로그인 정도. 나머지는 스크립트가 처리 |

## 준비물

- 노트북 PC (macOS / Windows / Linux) + **VS Code**
- AWS 계정 (강사가 제공하는 계정이나 개인 계정). g6e 인스턴스 쿼터 16 vCPU 이상
- 인터넷 (Hugging Face 와 GitHub 에서 다운로드)

## 모듈 구성

| # | 모듈 | 시간 | 내용 |
|---|---|---|---|
| 0 | [강사용 가이드](00_facilitator.md) | — | 사전 준비, 비용, 운영 팁, 실습 정답 |
| 1 | [개념: LLM 에 지식을 넣는 방법](01_concepts.md) | 40분 | 사전학습 / CPT / SFT / RAG, LoRA vs full, 망각, perplexity |
| 2 | [실습 환경 구성](02_environment.md) | 40분 | AWS g6e.4xlarge 생성, VS Code Remote-SSH, Jupyter 커널 |
| 3 | [데이터 파이프라인](03_data.md) | 50분 | 수집 → 정제 → 토큰 패킹 → replay 혼합 |
| 4 | [계속 사전학습](04_training.md) | 60분 | 학습 코드 분석, 메모리 계산, 학습 실행·모니터링 |
| 5 | [평가](05_evaluation.md) | 40분 | perplexity 3종, 이어쓰기 프로브, 사실 검증 |
| 6 | [다음 단계와 정리](06_next_steps.md) | 30분 | SFT, Chat Vector, RAG, 서빙, 리소스 정리 |
| 부록 | [문제 해결 · FAQ · 용어집](99_troubleshooting.md) | — | 자주 나는 오류와 해결법 |

전체 약 **4시간 20분** (쉬는 시간 제외). 시간표 예시는 [강사용 가이드](00_facilitator.md#시간표-예시) 참고.

## 프로젝트 파일 지도

```
llm-ft/
├── workshop/                 ← 지금 보고 있는 워크숍 문서
├── AWS_G6E_GUIDE.md          ← AWS 환경 요약 (모듈 2 의 축약판)
│
├── 01_download.sh            ① 문서 수집 (sparse checkout)
├── 02_prepare_data.py        ① 정제 함수 + (선택) SFT 데이터
├── 03_build_cpt.py           ① 토큰 패킹 + replay → data/cpt, data/eval
├── train_cpt_hf.py           ② CUDA 학습 (full / LoRA)
├── k8s_cpt_g6e.ipynb         ②③ 실습 노트북 (g6e 에서 실행)
│
├── aws/
│   ├── launch_g6e.sh         인스턴스 생성 + ~/.ssh/config 등록
│   ├── setup_remote.sh       원격 Python 환경 + Jupyter 커널
│   ├── sync.sh               코드 업로드 / 결과 회수
│   └── ctl.sh                start / stop / status / terminate / allow-my-ip
│
├── config_cpt.yaml           (Mac 대안) MLX 로 CPT
├── 05_eval_cpt.py            (Mac 대안) MLX 로 평가
├── train_hf.py, config_lora.yaml, 04_compare.py   (모듈 6) SFT
└── requirements-cuda.txt, requirements-mac.txt
```

## 라이선스와 출처

- 학습 데이터: [kubernetes/website](https://github.com/kubernetes/website) 문서, **CC BY 4.0**.
  학습한 모델이나 데이터를 배포할 때는 출처를 표기해야 한다.
- 일반 텍스트 replay: [wikimedia/wikipedia](https://huggingface.co/datasets/wikimedia/wikipedia) `20231101.ko`, CC BY-SA 4.0.
- 기본 모델: [Qwen/Qwen2.5-3B](https://huggingface.co/Qwen/Qwen2.5-3B). 3B 모델은 **Qwen Research License** 라서
  상업적으로 이용하려면 라이선스를 확인해야 한다. 대안 모델은 모듈 1 에 정리했다.
