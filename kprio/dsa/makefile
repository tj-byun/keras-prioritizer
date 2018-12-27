all:
	swig -python -c++ -o _dsa_module.cc dsa.i
	python setup.py build_ext --inplace
	cp _dsa.so ../
	cp dsa.py ../

clean:
	rm *.cc *.so ../_dsa.so ../dsa.py
