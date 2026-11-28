"""Interactive trace viewer. Geometry is illustrative; physics lives in joint_bench."""
import math


def show(trace):
    import vpython as vp
    v = vp.vector
    scene = vp.canvas(title='Front-leg joint bench', width=980, height=490,
                      background=v(.045, .063, .080), center=v(0, -.06, 0),
                      forward=v(-.10, -.12, -1), range=1.45)
    scene.autoscale = False
    scene.append_to_caption('Drag to rotate · scroll to zoom. Tendon thickness shows tension.\n')
    rigs = []
    names = list(trace['trials'])
    for i, name in enumerate(names):
        pivot = v((i-(len(names)-1)/2)*1.25, .2, 0)
        vp.cylinder(pos=pivot+v(-.38,.5,0), axis=v(.38,-.5,0), radius=.065,
                    color=v(.70,.64,.49))
        vp.sphere(pos=pivot, radius=.09, color=v(.82,.85,.87))
        shin = vp.cylinder(pos=pivot, radius=.054, color=v(.80,.73,.55))
        tendons = [vp.cylinder(pos=pivot+v(s*.12,.26,.08), radius=.014, color=color)
                   for s,color in ((1,v(1,.47,.19)),(-1,v(.18,.74,.82)))]
        stop_angle = trace['stop_rad']
        stop = vp.sphere(pos=pivot+v(math.sin(stop_angle),-math.cos(stop_angle),0)*.62,
                         radius=.08, color=v(.9,.31,.32), opacity=.6)
        vp.label(pos=pivot+v(0,.70,0), text=name, height=14, box=False, opacity=0)
        rigs.append((pivot, shin, tendons, stop))
    frame = 0
    last = len(trace['trials'][names[0]])-1
    playing = True

    def toggle(button):
        nonlocal playing
        playing = not playing
        button.text = 'Pause' if playing else 'Play'

    def seek(control):
        nonlocal frame, playing
        frame = int(control.value)
        playing = False
        play_button.text = 'Play'

    play_button = vp.button(text='Pause', bind=toggle)
    slider = vp.slider(min=0, max=last, step=1, value=0, bind=seek, length=370)
    clock = vp.wtext(text='')
    while True:
        vp.rate(50)
        for name, (pivot, shin, tendons, stop) in zip(names, rigs):
            row = trace['trials'][name][frame]
            angle = row['angle']
            axis = v(math.sin(angle), -math.cos(angle), 0)
            normal = v(math.cos(angle), math.sin(angle), 0)
            shin.axis = .62*axis
            for sign, key, line in zip((1,-1), ('forward','reverse'), tendons):
                line.axis = pivot + .23*axis + sign*.08*normal + v(0,0,.08) - line.pos
                line.radius = .009 + .003*row[key]
            stop.visible = row['obstacle']
        clock.text = f" {row['t']:.3f} s · drive {row['command']:.2f}"
        if playing:
            slider.value = frame
            frame = (frame+4) % (last+1)
