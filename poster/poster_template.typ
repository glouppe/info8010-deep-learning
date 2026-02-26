#import "@preview/tiaoma:0.3.0": barcode
#import "@preview/cetz:0.4.2": canvas, draw

// Document
#set document(title: "A Minimal Research Poster Template in Typst", author: "François Rozet")

// Text
#set par(justify: true)
#set text(font: "Fira Sans", size: 8pt, weight: "light", hyphenate: false)
#set strong(delta: 200)
#set underline(offset: 1em)

// Equations
#set math.equation(numbering: "(1)")

// Headings
#show heading: set text(size: 1em, weight: "medium")
#show heading: set block(above: 1.5em, below: 1em)

// Lists
#set list(tight: false, marker: strong("•"))
#set enum(tight: false)

// Figures
#set figure.caption(separator: [. ])

// Macros
#let grad(x) = $nabla_(#h(-0.25em) #x)$
#let showcase(inset: 1em, body) = {
    rect(fill: rgb("#f0f0f0"), inset: inset, radius: inset, width: 100%, body)
}
#let social(url, logo, body) = {
    link(url, stack(dir: ltr, spacing: 0.5em, image(logo, height: 1.25em), text(size: 1.25em, body)))
}
#let theorem(body) = [
    #counter("theorem").step() *Theorem #context counter("theorem").display().* #body
]

// Page
#set page(paper: "a4", flipped: true, margin: 0.5cm, columns: 3)
#set columns(gutter: 0.5cm)

// Title
#place(
    top,
    float: true,
    clearance: 0.5cm,
    scope: "parent",
    stack(
        dir: ltr,
        spacing: 1fr,
        // Title
        align(
            horizon,
            stack(
                dir: ttb,
                spacing: 1em,
                text(size: 2.0em, weight: "medium", [A Minimal Research Poster Template in Typst]),
                text(size: 1.25em, [François Rozet, Fanny Bodart and Sheldon Cooper]),
            )
        ),
        // Logos
        align(horizon, image("figures/typst.svg", height: 0.8cm)),
        align(horizon, image("figures/ferris.svg", height: 0.8cm)),
        align(horizon, image("figures/rust.svg", height: 0.8cm)),
        // Links
        align(horizon + center, stack(
            dir: ttb,
            spacing: 0.75em,
            social("https://arxiv.org/abs/2304.00319", "figures/arxiv.svg", `arXiv:2304.00319`),
            social("https://github.com/francois-rozet/postr", "figures/github.svg", `francois-rozet/postr`),
        )),
        link(
            "https://youtu.be/dQw4w9WgXcQ",
            barcode(
                "https://youtu.be/dQw4w9WgXcQ",
                "QRCode",
                height: 1.5cm,
            ),
        ),
    ),
)

// Body

= Abstract

#lorem(42)

$ p(x_(1:L) | y) = p(y | x_(1:L)) / p(y) p(x_1) product_(i=1)^(L-1) p(x_(i+1) | x_i) $

#lorem(30)

#figure({
    set math.equation(numbering: none)

    canvas(length: 3em, {
        draw.set-style(
            circle: (fill: rgb("#f0f0f0"), stroke: none),
            mark: (scale: 0.5, fill: black),
            stroke: (thickness: 0.075em),
        )

        draw.circle((-1, 0), radius: 0.4, name: "x1")
        draw.content("x1", $ x_1 $)
        draw.line((-0.5, 0), (1.5, 0), mark: (end: ">"))

        draw.circle((2, 0), radius: 0.4, name: "x2")
        draw.content("x2", $ x_2 $)
        draw.line((2.5, 0), (3.5, 0), mark: (end: ">"))

        draw.circle((4, 0), radius: 0.4, name: "x3")
        draw.content("x3", $ dots $)
        draw.line((4.5, 0), (5.5, 0), mark: (end: ">"))

        draw.circle((6, 0), radius: 0.4, name: "x4")
        draw.content("x4", $ x_L $)

        draw.circle((3, -1.5), radius: 0.4, name: "y")
        draw.content("y", $ y $)

        for x in ("x1", "x2", "x3", "x4") {
            draw.line((a: x, b: "y", number: 0.5, abs: true), (a: "y", b: x, number: 0.5, abs: true), mark: (end: ">"))
        }

        draw.content((0.5, 0.25), $ p(x_(i+1) | x_i) $, anchor: "south")
        draw.content((4, -1.5), $ p(y | x_(1:L)) $, anchor: "west")
    })
})
= Problem Statement
#lorem(81)

#showcase(theorem(lorem(42)))

#lorem(55)

#colbreak()

= Methods \& contributions

#[
#set enum(
    numbering: i => rect(
        fill: yellow,
        inset: 1em,
        radius: 1em,
        text(size:1.5em, fill: black, strong([#i]))
    )
)

1. #showcase([Lorem ipsum dolor sit amet, consectetur adipiscing elit, sed do *eiusmod tempor incididunt* ut labore et dolore magnam aliquam quaerat voluptatem.])

#lorem(12)

2. #showcase([Ut enim aeque doleamus animo, cum corpore dolemus, ﬁeri tamen *permagna accessio potest*, si aliquod aeternum et inﬁnitum impendere malum nobis opinemur.])

#lorem(23)

#showcase(
```python
def fast_fibonacci(n: int) -> int:
    a, b = 0, 1

    for i in range(n):
        a, b = b, a + b

    return a
```
)



3. #showcase([Athenis, ut e patre audiebam facete et urbane Stoicos irridente, statua est in quo a nobis philosophia $a^2 + b^2 = c^2$ defensa et collaudata est, cum id, quod maxime $exp(i pi) + 1 = 0$ placeat, facere possimus, omnis voluptas assumenda est.])

#figure(image("figures/pseudo_blanket.pdf"), caption: [#lorem(13)])

]

#colbreak()

= Results

#lorem(30)

#figure(image("figures/navier-stokes.pdf"), caption: [#lorem(13)])

#lorem(15)

1. #lorem(20)
#figure(image("figures/rb_vrmse.svg"), caption: [#lorem(13)])
2. #lorem(40)

3. #lorem(12)

#lorem(11)

== Conclusion

I would like to thank the Typst team for this wonderful software. With Typst, typesetting is fast, intuitive and fun.
