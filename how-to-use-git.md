# Rules

- 개인 작업 내용은 'personal' branch 에 올려주세요.
- main branch 는 협업이 필요하거나 특정 버전의 결과물을 배포할 때 사용하려고 합니다.

---

# Git 사용법

```bash
# 먼저 자신의 git 폴더로 이동하여 우클릭 -> git bash를 선택하여 bash 창을 띄웁니다.
$ git status # check your git folder
```

```bash
# 다른 사람의 수정 사항을 현재 내 컴퓨터에 반영할 때, 내 컴퓨터의 파일을 최신 버전으로 업데이트
$ git pull

# 내 수정사항을 서버로 보내는 방법
$ git add .
$ git commit -m "<메시지>"
$ git push origin <branch, 우리는 personal을 주로 사용할 것>
```

아마 처음 사용할 때 `push` 명렁어 사용 시 에러가 뜰 수 있는데, 이 때는 아래 코드를 실행해주세요.

```bash
$ git config --global user.name "Your Name"        # github nickname
$ git config --global user.email you@example.com   # github email
```

아래 명령어를 사용하면 한 번만 로그인하면 이후에 `push` 명령어를 사용할 때 다시 로그인할 필요가 없습니다.

```
git config credential.helper store
```