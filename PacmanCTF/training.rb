10000.times do |i|
  puts "Iterated #{i * 20} time(s)"
  puts `pypy capture.py -r betaqTeam -q -b alphaTeam  -n 20 -l tinyCapture`
  puts `pypy capture.py -r betaqTeam -q  -n 20`
  puts `pypy capture.py -r betaqTeam -q -b alphaTeam  -n 20`
end
