virtualenv -p /var/lang/bin/python3.6 --always-copy --no-site-packages venv

source venv/bin/activate

pip3 install --upgrade pip wheel
#pip3 install numpy scipy Pillow redis requests

libdir="$VIRTUAL_ENV/lib/python3.6/site-packages/"
mkdir -p $libdir/lib || true
echo "venv original size $(du -sh $VIRTUAL_ENV | cut -f1)"
#find $libdir -name "tests" | xargs rm -r
# Manually install h5py as it has tests
pip install h5py
find $libdir -name "dataset" | xargs rm -rf
find $libdir -name "*.pyc" -delete
find $libdir -type d -empty -delete
find $libdir -name "*.so" | xargs strip
echo "venv new size $(du -sh $VIRTUAL_ENV | cut -f1)"
pushd $VIRTUAL_ENV/lib/python3.6/site-packages/
zip -r -9 /var/task/package.zip *
popd
echo "venv original size $(du -sh $VIRTUAL_ENV | cut -f1)"
deactivate

cd /var/task/src
zip -r9 ../package.zip *.py
#cd ..
#rm -rf venv