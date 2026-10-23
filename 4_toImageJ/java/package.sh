#!/usr/bin/env bash
# One-click build: bundle a .npy LUT into Seg-LUT.jar for Fiji (ImageJ 1.x).
# Usage:
#   ./package.sh
#   ./package.sh /path/to/Seg2_LUT_HV_2_4.npy

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
LUT_PATH="${1:-${SCRIPT_DIR}/../../3_Test_using_LUT/Seg2_LUT_HV_2_4.npy}"
RESOURCE_DIR="${SCRIPT_DIR}/src/main/resources/lut"
DEST="${RESOURCE_DIR}/Seg2_LUT_HV_2_4.npy"

mkdir -p "${RESOURCE_DIR}"
cp -f "${LUT_PATH}" "${DEST}"
echo "Bundled LUT: ${DEST}"

cd "${SCRIPT_DIR}"
if command -v mvn >/dev/null 2>&1; then
  mvn -q clean package
elif [[ -x "${SCRIPT_DIR}/mvnw" ]]; then
  "${SCRIPT_DIR}/mvnw" -q clean package
else
  echo "Maven not found. Install Maven or add mvnw (Unix)." >&2
  exit 1
fi
mkdir -p "${SCRIPT_DIR}/../dist"
cp -f target/Seg-LUT.jar "${SCRIPT_DIR}/../dist/Seg-LUT.jar"
echo ""
echo "Built: ${SCRIPT_DIR}/../dist/Seg-LUT.jar"
echo "Install: copy Seg-LUT.jar to Fiji.app/plugins/ and restart Fiji."
echo "Menu: Plugins -> Segmentation -> SEG-LUT"
