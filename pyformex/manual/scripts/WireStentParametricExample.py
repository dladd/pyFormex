from examples.WireStent import DoubleHelixStent

for i in [16.,32]:
    for j in [6,10]:
        for k in [25,50]:
            stent = DoubleHelixStent(i,40.,0.22,j,k).all()
            draw(stent,view='iso')
            pause()
            clear()
