# shellcheck disable=SC1113
#/bin/bash
die () {
	echo > "$@"
	exit 1
}

pushd .

python3 ./setup.py bdist_wheel || die 'python setup failed'
# shellcheck disable=SC2164
popd

echo; echo