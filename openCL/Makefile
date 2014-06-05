all: clDeviceQuery,saxpy,saxpy2,squaredCL

clDeviceQuery: clDeviceQuery.cpp Makefile
	g++ clDeviceQuery.cpp -lcl -o clDeviceQuery -I/usr/include -L/usr/lib/beignet

saxpy: saxpy.cpp Makefile
	g++ saxpy.cpp -lcl -o saxpy -I/usr/include -L/usr/lib/beignet

saxpy2: saxpy2.cpp Makefile
	g++ saxpy2.cpp -lcl -o saxpy2 -I/usr/include -L/usr/lib/beignet

squaredCL: squaredCL.cpp Makefile
	g++ squaredCL.cpp -lcl -o squaredCL -I/usr/include -L/usr/lib/beignet

clean:
	rm -f clDeviceQuery
	rm -f saxpy
	rm -f squaredCL