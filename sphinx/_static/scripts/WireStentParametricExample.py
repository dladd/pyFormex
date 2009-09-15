from examples.WireStent import DoubleHelixStent

for De in [16.,32.]:
    for nx in [6,10]:
        for beta in [25,50]:
            stent = DoubleHelixStent(De,40.,0.22,nx,beta).all()
            draw(stent,view='iso')
            pause()
            clear()
            
