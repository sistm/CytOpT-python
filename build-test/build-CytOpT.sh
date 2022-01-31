# shellcheck disable=SC1113
#/bin/bash
die () {
	echo >&2 "$@"
	exit 1
}

pushd .
cd ../CytOpT

python3 ../setup.py bdist_wheel || die 'python setup failed'
popd

echo; echo