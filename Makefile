make:
	clang++ Sandbox.cc -g -std=c++17 -Wall -Wextra -Wpedantic -o Sandbox.out

tests:
	clang++ TestSuite.cc -g -std=c++17 -Wall -Wextra -Wpedantic -o Tests.out

clean:
	ls -a | grep .raw | xargs rm
	rm a.out
