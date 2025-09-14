.PHONY: build clean test

build:
	go build -o stringart main.go

clean:
	rm -f stringart

test: build
	./stringart -input ae300.jpg -pins 400 -lines 20000 -output-size 2000 -weight 4 -output-weight 15
	open output.png
