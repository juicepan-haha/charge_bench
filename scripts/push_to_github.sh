#!/usr/bin/env bash
# 推送到 GitHub 并做推送后校验。
#
# 前置（只需做一次）：
#   1. 把 SSH 公钥加到 GitHub：Settings → SSH and GPG keys → New SSH key
#      cat ~/.ssh/id_ed25519_chargebench.pub
#   2. 在 GitHub 上创建一个**空**仓库（不要勾 README / .gitignore / license）
#
# 然后运行：bash scripts/push_to_github.sh
set -euo pipefail

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"

cd "$(dirname "$0")/.."

url="$(git remote get-url "$REMOTE" 2>/dev/null || true)"
if [ -z "$url" ]; then
  echo "✗ 没有配置 $REMOTE 远端。先执行：" >&2
  echo "    git remote add origin git@github.com:<你的用户名>/<仓库名>.git" >&2
  exit 1
fi
echo "远端: $url"

echo -n "检查 SSH 授权… "
if ! ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 | grep -q "successfully authenticated"; then
  echo "未通过" >&2
  echo "✗ SSH 公钥还没加到 GitHub，或本仓库的 core.sshCommand 未生效。" >&2
  echo "  待添加的公钥：" >&2
  cat "${HOME}/.ssh/id_ed25519_chargebench.pub" >&2 2>/dev/null || \
    echo "  （找不到 ~/.ssh/id_ed25519_chargebench.pub）" >&2
  exit 1
fi
echo "通过"

echo -n "检查远端仓库是否可写… "
if ! git ls-remote --exit-code "$REMOTE" >/dev/null 2>&1; then
  echo "不可访问" >&2
  echo "✗ 远端仓库可能还不存在。请在 GitHub 上创建一个**空**仓库（不勾任何初始化文件）后重试。" >&2
  exit 1
fi
echo "可访问"

echo -n "推送 $BRANCH… "
git push -u "$REMOTE" "$BRANCH"
echo "完成"

echo
echo "校验："
echo "  本地 HEAD   $(git rev-parse --short HEAD)  $(git log -1 --pretty=%s)"
echo "  远端 $BRANCH $(git rev-parse --short "$REMOTE/$BRANCH" 2>/dev/null || echo '(未跟踪)')"
echo "  提交数      $(git rev-list --count HEAD)"
echo "  工作区      $(git status --porcelain | wc -l) 个未提交改动"
echo
echo "✓ 若上面本地与远端的短哈希一致，推送成功。"
