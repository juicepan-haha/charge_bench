#!/usr/bin/env bash
# 推送并做推送后校验。
#
# 支持两种授权方式，脚本会自行识别：
#   A. gh CLI（HTTPS，推荐）—— gh auth login 之后即可，凭据由 gh 作为 git credential helper 提供
#   B. SSH 密钥 —— 适合不用 gh 的场景
#
# 用法：
#   bash scripts/push_to_github.sh
#   REMOTE=upstream BRANCH=main bash scripts/push_to_github.sh
set -euo pipefail

REMOTE="${REMOTE:-origin}"
BRANCH="${BRANCH:-main}"

cd "$(dirname "$0")/.."

url="$(git remote get-url "$REMOTE" 2>/dev/null || true)"
if [ -z "$url" ]; then
  echo "✗ 没有配置 $REMOTE 远端。先执行：" >&2
  echo "    git remote add origin <仓库地址>" >&2
  exit 1
fi
echo "远端: $url"
echo "分支: $BRANCH（本地 HEAD $(git rev-parse --short HEAD)）"

# 有未提交改动时先提醒 —— 推送不会带上它们，容易误以为已经上去了
dirty="$(git status --porcelain | wc -l)"
if [ "$dirty" -ne 0 ]; then
  echo "⚠ 工作区有 $dirty 个未提交改动，**不会**被推送。先提交再推。" >&2
fi

echo -n "检查授权… "
case "$url" in
  git@*|ssh://*)
    if ssh -o BatchMode=yes -o StrictHostKeyChecking=accept-new -T git@github.com 2>&1 \
        | grep -q "successfully authenticated"; then
      echo "SSH 通过"
    else
      echo "未通过" >&2
      echo "✗ SSH 公钥未授权。把下面这行加到 GitHub → Settings → SSH and GPG keys：" >&2
      cat "${HOME}/.ssh/id_ed25519_chargebench.pub" >&2 2>/dev/null \
        || echo "  （找不到 ~/.ssh/id_ed25519_chargebench.pub）" >&2
      echo "  注意：必须复制**一整行**，别把行首行尾的装饰符或换行一起带上。" >&2
      exit 1
    fi
    ;;
  https://*)
    if ! command -v gh >/dev/null 2>&1; then
      echo "无 gh" >&2
      echo "✗ HTTPS 推送需要凭据。装 gh 后执行 gh auth login，或改用 SSH 远端。" >&2
      exit 1
    fi
    if ! gh auth status >/dev/null 2>&1; then
      echo "未登录" >&2
      echo "✗ gh 未登录。执行：gh auth login --hostname github.com --git-protocol https --web" >&2
      exit 1
    fi
    # gh auth setup-git 会把 gh 注册成 git 的凭据助手；缺了它 git 会报 "could not read Username"
    if ! git config --get-all credential.https://github.com.helper >/dev/null 2>&1; then
      echo -n "补配凭据助手… "
      gh auth setup-git --hostname github.com
    fi
    echo "gh 通过（$(gh auth status 2>&1 | grep -oE 'account [^ ]+' | head -1)）"
    ;;
  *)
    echo "未知协议，跳过检查" ;;
esac

echo -n "检查远端仓库可访问… "
if ! git ls-remote --exit-code "$REMOTE" >/dev/null 2>&1; then
  echo "不可访问" >&2
  echo "✗ 远端仓库不存在或无权访问。私有仓库请确认已创建且账号有权限。" >&2
  exit 1
fi
echo "可访问"

echo -n "推送… "
git push -u "$REMOTE" "$BRANCH"
echo "完成"

echo
echo "校验："
echo "  本地 HEAD      $(git rev-parse --short HEAD)  $(git log -1 --pretty=%s)"
echo "  远端 $BRANCH$(printf '%*s' $((7 - ${#BRANCH})) '') $(git rev-parse --short "$REMOTE/$BRANCH" 2>/dev/null || echo '(未跟踪)')"
echo "  提交数         $(git rev-list --count HEAD)"
echo "  远端点对象数   $(git ls-tree -r "$REMOTE/$BRANCH" --name-only 2>/dev/null | wc -l)"
echo
if [ "$(git rev-parse HEAD)" = "$(git rev-parse "$REMOTE/$BRANCH" 2>/dev/null || echo '')" ]; then
  echo "✓ 本地与远端哈希一致，推送成功。"
else
  echo "✗ 哈希不一致，请检查上面的输出。" >&2
  exit 1
fi
