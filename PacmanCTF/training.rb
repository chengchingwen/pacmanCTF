10000.times do |i|
  puts "Iterated #{i * 20} time(s)"
  puts `pypy capture.py -r comTeam -q -b alphaTeam  -n 20 -l tinyCapture`
end
