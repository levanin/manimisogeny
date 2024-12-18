from manim import *

class LabeledDot(Dot):
    def __init__(self, label, radius=None, **kwargs) -> None:
        if isinstance(label, str):
            rendered_label = MathTex(label, color=BLACK)
        else:
            rendered_label = label

        if radius is None:
            radius = (
                0.1 + max(rendered_label.get_width(), rendered_label.get_height()) / 2
            )
        Dot.__init__(self, radius=radius, **kwargs)
        rendered_label.move_to(self.get_center())
        self.add(rendered_label)

class JText(Text):
    CONFIG = {
        'font': 'Linux Libertine Display', #'Nimbus Roman',
        'size': 0.5,
        }

class EC(VMobject):
    '''
    A short Weierstrass curve with a controlled zero:

        y² = (x - a)(x² + ax + c)
    '''
    
    def __init__(self, a, c, **kwargs):
        super().__init__(**kwargs)
        self.a = a
        self.c = c
        self.disc = a*a - 4*c
        self.zeros = [a]
        if self.disc == 0:
            self.zeros.append(-a/2)
        elif self.disc > 0:
            self.zeros.append((-a + np.sqrt(self.disc))/2)
            self.zeros.append((-a - np.sqrt(self.disc))/2)
        self.zeros.sort()

    def plot(self, x_min, x_max, graph, **kwargs):
        if len(self.zeros) < 3:
            xmin = min(max(self.zeros[0], x_min), x_max)
            xmax = x_max
            self.add(ParametricFunction(self.branch(xmin, xmax, graph, False), **kwargs))
            self.add(ParametricFunction(self.branch(xmin, xmax, graph, True), **kwargs))
        else:
            x0 = min(max(self.zeros[0], x_min), x_max)
            x1 = min(max(self.zeros[1], x_min), x_max)
            x2 = min(max(self.zeros[2], x_min), x_max)
            xmax = x_max
            self.add(ParametricFunction(self.branch(x2, xmax, graph, False), **kwargs))
            self.add(ParametricFunction(self.branch(x2, xmax, graph, True), **kwargs))
            self.add(ParametricFunction(self.branch(x0, x1, graph, False), **kwargs))
            self.add(ParametricFunction(self.branch(x0, x1, graph, True), **kwargs))
        return self
        
    def branch(self, xmin, xmax, graph, neg=False):
        def func(t):
            x = interpolate(xmin, xmax, t if neg else 1-t)
            y2 = (x - self.a)*(x*x + self.a*x + self.c)
            y = (-1)**neg * np.sqrt(y2) if y2 > 0 else 0
            return graph.coords_to_point(x, y)
        return func

    def get_y(self, x):
        y2 = (x - self.a)*(x*x + self.a*x + self.c)
        return np.sqrt(y2) if y2 >= 0 else None

    def group_law(self, x0, y0, x1, y1):
        λ = (y0-y1)/(x0-x1)
        x = λ**2 - x0 - x1
        y = -y0 - λ*(x - x0)
        return (x, y)


# Tex.CONFIG['color'] = BLACK
# MathTex.CONFIG['color'] = BLACK
# Mobject.CONFIG['color'] = BLACK
# Dot.CONFIG['color'] = BLACK

class GroupLaw(Scene):
    
    def construct(self):
        # draw axes
        x_min = -10
        x_max = 10
        axes = Axes(
            x_range=[x_min, x_max],
            y_range=[-10, 10],
            axis_config=dict(
                stroke_color=GREY_A,
                stroke_width=2,
                numbers_to_exclude=[0],
            )
        )
        self.add(axes)

        # draw elliptic curve
        ec = EC(-6, 10).plot(x_min, x_max, axes, color=BLACK)
        self.add(ec)
        self.wait()

        # draw group law
        Py = ec.get_y(-4)
        Qy = ec.get_y(-1)
        Sx, Sy = ec.group_law(-4, Py, -1, Qy)
        P = Dot(axes.coords_to_point(-4, Py), color=BLACK)
        Q = Dot(axes.coords_to_point(-1, Qy), color=BLACK)
        R = Dot(axes.coords_to_point(Sx, -Sy), color=BLACK)
        S = Dot(axes.coords_to_point(Sx, Sy), color=BLACK)
        Plab = MathTex('P',color=BLACK).next_to(P, UP)
        Qlab = MathTex('Q', color = BLACK).next_to(Q, UP)
        Slab = MathTex('P+Q', color = BLACK).next_to(S, RIGHT)

        self.play(FadeIn(P), FadeIn(Plab), FadeIn(Q), FadeIn(Qlab), runtime=2)
        self.wait(2)
        PQ = Line(P, Q, color=RED)
        PQ.set_length(10)
        self.play(Create(PQ), runtime=2)
        self.play(FadeIn(R))
        self.wait(2)
        RS = DashedLine(R, S, color=RED).scale(1.1)
        self.play(Create(RS), runtime=2)
        self.play(FadeIn(S), FadeIn(Slab))
        self.wait(2)
        self.play(FadeOut(PQ), FadeOut(R), FadeOut(S), FadeOut(Slab),FadeOut(P), FadeOut(Q), FadeOut(Plab), FadeOut(Qlab),FadeOut(RS), runtime=2)

        # draw multiplication by 2
        Mx = -6
        M = Dot(axes.coords_to_point(Mx,0), color=BLACK)
        Mlab = MathTex('R', color=BLACK).next_to(M, LEFT)
        self.play(FadeIn(M), FadeIn(Mlab), runtime=2)
        self.wait(2)
        Mp = Dot(axes.coords_to_point(Mx,1), color=BLACK)
        MM = Line(M, Mp, color=RED)
        MM.set_length(20)
        self.play(Create(MM), runtime=2)
        MplusMlab1 = MathTex('R+R', color=BLACK).next_to(axes.coords_to_point(Mx,20), DR)
        MplusMlab2 = MathTex('2R', color=BLACK).next_to(axes.coords_to_point(Mx,20), DR)
        MplusMlab3 = MathTex('\infty', color=BLACK).next_to(axes.coords_to_point(Mx,20), DR)
        self.play(Write(MplusMlab1), runtime=2)
        self.wait(1)
        self.play(Transform(MplusMlab1,MplusMlab2), runtime=2)
        self.wait(1)
        self.play(Transform(MplusMlab1, MplusMlab3), runtime=2)
        self.wait(2)

        
        
class Isomorphism(CoordinateSystem):
    CONFIG = {
        'x_min': -10, 'x_max': 10, 'y_min': -10, 'y_max': 10,
        'graph_origin': ORIGIN, 'function_color': BLACK, 'axes_color': GRAY,
        'guides_color': RED
        }

    def construct(self):
        # draw axes
        self.setup_axes(animate=False)
        # draw elliptic curve
        ec = Group(
            Line(self.coords_to_point(0,-10), self.coords_to_point(0,10), color=BLACK),
            Line(self.coords_to_point(-10,0), self.coords_to_point(10,0), color=BLACK),
            EC(-4, 7).plot(self.x_min, self.x_max, self, color=BLACK),
        )
        self.play(FadeIn(ec))
        self.wait()

        # Transform EC
        self.play(ec.move_to, [.5,.5,0], run_time=2)
        self.play(ec.scale, 1.5, run_time=2)
        self.play(ApplyMatrix([[1,1],[0,1]], ec), run_time=2)
        self.play(ec.move_to, [-2,-2,0], run_time=2)
        self.play(ec.scale, 0.02, FadeOut(self.axes), run_time=2)
        # Write j-invariant formula
        j = MathTex(r'\quad j(\quad\quad) =').move_to(ec)
        formula = MathTex(r'1728\frac{4a^3}{4a^3+27b^2}').next_to(j, RIGHT)
        self.play(FadeIn(j), FadeIn(formula))
        self.wait()

        # Squash to a graph node and show graph of GF(71)
        js = [
            LabeledDot('24', color=RED).move_to(formula),
            LabeledDot('41', color=RED).move_to([-2,0,0]),
            LabeledDot('0', color=RED).move_to([4,-2,0]),
            LabeledDot('40', color=RED).move_to([2,0,0]),
            LabeledDot('17', color=RED).move_to([-4,-2,0]),
            LabeledDot('48', color=RED).move_to([4,2,0]),
            LabeledDot('66', color=RED).move_to([-4,2,0]),
        ]
        self.play(FadeOut(formula), FadeIn(js[0]))
        self.wait()

        # Populate isogeny graph
        self.play(FadeOut(j), FadeOut(ec))
        for j in js[1:]:
            self.play(FadeIn(j), run_time=0.5)
        edges = [
            CubicBezier([js[0].get_critical_point(DL),
                         js[0].get_critical_point(DL) + [-1,-1,0],
                         js[0].get_critical_point(DR) + [1,-1,0],
                         js[0].get_critical_point(DR)]),
            Line(js[0], js[4]),
            Line(js[1], js[4]),
            Line(js[1], js[6]),
            Line(js[2], js[3]),
            Line(js[3], js[4]),
            Line(js[3], js[5]),
            CubicBezier([js[5].get_critical_point(UL),
                         js[5].get_critical_point(UL) + [-1,1,0],
                         js[5].get_critical_point(UR) + [1,1,0],
                         js[5].get_critical_point(UR)]),
            Line(js[5], js[6]),
        ]
        for e in edges:
            self.play(Create(e))
        
        self.wait(2)
