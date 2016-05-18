10000.times do |i|
  puts "Iterated #{i * 20} time(s)"
  puts `pypy capture.py -r simpleAgent -q -b alphaTeam  -n 20 -l tinyCapture`
  puts `pypy capture.py -r simpleAgent -q  -n 20`
  puts `pypy capture.py -r simpleAgent -q -b alphaTeam  -n 20`
end
