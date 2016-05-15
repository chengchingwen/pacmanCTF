10000.times do |i|
  puts "Iterated #{i * 20} time(s)"
  puts `pypy capture.py -b comTeam -r alphaTeam  -q -n 20`
  puts `pypy capture.py -b comTeam  -q -n 20`
end
