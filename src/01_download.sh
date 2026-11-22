#!/usr/bin/env bash
# 쿠버네티스 공식 웹사이트 저장소에서 문서만 sparse checkout 으로 받는다.
# 문서 라이선스: CC BY 4.0 (https://github.com/kubernetes/website)
#   ./01_download.sh            # 한글만
#   LANGS="ko en" ./01_download.sh  # 영어 원문도 (한글 번역은 원문의 일부만 다룸)
set -euo pipefail
LANGS=${LANGS:-ko}

mkdir -p raw
if [ ! -d raw/website ]; then
  git clone --depth 1 --filter=blob:none --sparse \
    https://github.com/kubernetes/website.git raw/website
fi
# 문서 본문(content/<언어>) + 정제에 필요한 사이트 자원:
#   content/en/examples   한글에 없는 예제 YAML 은 사이트도 영어판을 쓴다 (code_sample 숏코드)
#   content/en/.../feature-gates   기능 게이트별 단계·버전 정보 (feature-state 숏코드)
#   i18n                  숏코드 번역 문자열 (heading, feature-state)
#   data/releases         패치 버전 정보 (skew currentPatchVersion)
#   hugo.toml (루트 파일은 항상 포함)   현재 버전 (skew, param)
# shellcheck disable=SC2046
git -C raw/website sparse-checkout set $(printf 'content/%s ' $LANGS) \
  content/en/examples \
  content/en/docs/reference/command-line-tools-reference/feature-gates \
  i18n data/releases

for l in $LANGS; do
  echo "[$l] 마크다운 문서 수: $(find raw/website/content/$l/docs -name '*.md' | wc -l)"
done
