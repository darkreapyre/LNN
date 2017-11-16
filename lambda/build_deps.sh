virtualenv -p /usr/local/bin/python --always-copy --no-site-packages deps

source deps/bin/activate

pip install --upgrade pip wheel
pip install -r requirements.txt

libdir="$VIRTUAL_ENV/lib/python3.6/site-packages/"
mkdir -p $libdir/lib || true
echo "venv original size $(du -sh $VIRTUAL_ENV | cut -f1)"
find $libdir -name "tests" | xargs rm -r
find $libdir -name "dataset" | xargs rm -rf
find $libdir -name "*.pyc" -delete
find $libdir -type d -empty -delete
find $libdir -name "*.so" | xargs strip
pushd $VIRTUAL_ENV/lib/python3.6/site-packages/
zip -r -9 deps.zip *
popd
deactivate

#rm -rf package

#cd src

#zip -r ../package.zip *.py