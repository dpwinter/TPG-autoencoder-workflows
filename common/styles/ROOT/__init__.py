# coding: utf-8

import ROOT
ROOT.PyConfig.IgnoreCommandLineOptions = True
ROOT.gROOT.SetBatch()


default_text_size = 22
default_canvas_width = 800
default_canvas_height = 640

default_style_props = {
    "OptStat": 0,
}

default_canvas_props = {
    "TopMargin": 0,
    "RightMargin": 0,
    "BottomMargin": 0,
    "LeftMargin": 0,
    "FillStyle": 1001,
}

default_pad_props = {
    "Pad": (0, 0, 1, 1),
    "TopMargin": 0.05,
    "RightMargin": 0.03,
    "BottomMargin": 0.105,
    "LeftMargin": 0.13,
    "FillStyle": 4000,
    "Ticks": (True, True),
}

default_axis_props = {
    "TitleFont": 43,
    "TitleSize": 25,
    "LabelFont": 43,
    "LabelSize": default_text_size,
}

default_auto_ticklength = 0.015

default_label_props = {
    "TextFont": 43,
    "TextSize": default_text_size,
    "TextAlign": 11,
    "NDC": True,
}

default_legend_props = {
    "BorderSize": 0,
    "FillStyle": 0,
    "FillColor": 0,
    "LineStyle": 0,
    "LineColor": 0,
    "LineWidth": 0,
    "TextFont": 43,
    "TextSize": default_text_size,
    "ColumnSeparation": 0.,
}

default_hist_props = {
    "LineWidth": 2,
    "LineColor": 1,
    "MarkerColor": 1,
}

default_graph_props = {
    "Title": "",
    "LineColor": 1,
    "LineWidth": 2,
    "FillColor": 0,
}

default_line_props = {
    "LineWidth": 2,
    "LineColor": 1,
    "NDC": True,
}

default_box_props = {
    "LineWidth": 2,
    "LineColor": 1,
    "FillColor": 0,
}

default_func_props = {
    "LineWidth": 2,
    "LineColor": 1,
}

colors = {
    "black": ROOT.kBlack,
    "blue": ROOT.kBlue + 1,
    "red": ROOT.kRed - 4,
    "magenta": ROOT.kMagenta + 1,
    "yellow": ROOT.kOrange - 2,
    "green": ROOT.kGreen + 2,
    "brightgreen": ROOT.kGreen - 3,
    "darkgreen": ROOT.kGreen + 4,
    "creamblue": 38,
    "creamred": 46,
    "white": 10,
    "ttH": ROOT.TColor.GetColor(67, 118, 201),
}


def setup_style(props=None):
    apply_root_properties(ROOT.gStyle, default_style_props, props)


def setup_canvas(canvas, width, height, props=None):
    canvas.SetWindowSize(width, height)
    canvas.SetCanvasSize(width, height)
    apply_root_properties(canvas, default_canvas_props, props)


def setup_pad(pad, props=None):
    apply_root_properties(pad, default_pad_props, props)


def setup_x_axis(x_axis, pad, props=None):
    canvas_height = pad.GetCanvas().GetWindowHeight()

    _props = default_axis_props.copy()

    # auto ticks
    pad_width = 1. - pad.GetLeftMargin() - pad.GetRightMargin()
    real_height = pad.YtoPixel(pad.GetY1()) - pad.YtoPixel(pad.GetY2())
    real_width = pad.XtoPixel(pad.GetX2()) - pad.XtoPixel(pad.GetX1())
    if pad_width != 0 and real_height != 0:
        _props["TickLength"] = default_auto_ticklength / pad_width * real_width / real_height

    _props["TitleOffset"] = 1.075 * default_canvas_height / canvas_height

    apply_root_properties(x_axis, _props, props)


def setup_y_axis(y_axis, pad, props=None):
    canvas_width = pad.GetCanvas().GetWindowWidth()

    _props = default_axis_props.copy()

    _props["TitleOffset"] = 1.4 * default_canvas_width / canvas_width

    # auto ticks
    pad_height = 1. - pad.GetTopMargin() - pad.GetBottomMargin()
    if pad_height != 0:
        _props["TickLength"] = default_auto_ticklength / pad_height

    apply_root_properties(y_axis, _props, props)


def setup_z_axis(z_axis, pad, props=None):
    canvas_width = pad.GetCanvas().GetWindowWidth()

    _props = default_axis_props.copy()

    _props["TitleOffset"] = 1.4 * default_canvas_width / canvas_width

    apply_root_properties(z_axis, _props, props)


def setup_label(label, props=None):
    apply_root_properties(label, default_label_props, props)


def calc_legend_pos(n_entries, x1=0.68, x2=0.96, y2=0.92, y_spread=0.045):
    y1 = y2 - y_spread * n_entries
    return (x1, y1, x2, y2)


def setup_legend(legend, props=None):
    apply_root_properties(legend, default_legend_props, props)


def get_canvas_pads(canvas):
    return [
        p for p in canvas.GetListOfPrimitives()
        if isinstance(p, ROOT.TPad)
    ]


def setup_hist(hist, props=None):
    apply_root_properties(hist, default_hist_props, props)


def setup_graph(graph, props=None):
    apply_root_properties(graph, default_graph_props, props)


def setup_line(line, props=None):
    apply_root_properties(line, default_line_props, props)


def setup_box(box, props=None):
    apply_root_properties(box, default_box_props, props)


def setup_func(func, props=None):
    apply_root_properties(func, default_func_props, props)


def update_canvas(canvas):
    for pad in get_canvas_pads(canvas):
        pad.RedrawAxis()
    ROOT.gPad.RedrawAxis()

    canvas.Update()


def apply_root_properties(obj, props, *_props):
    props = props.copy()
    for p in _props:
        if p:
            props.update(p)

    for name, value in props.items():
        setter = getattr(obj, "Set%s" % name, getattr(obj, name, None))
        if not hasattr(setter, "__call__"):
            continue

        if isinstance(value, tuple):
            # value might be a tuple (or list) of (class name, ...)
            # in that case, pass ROOT.<className>(*value[1:]) to the setter
            if isinstance(value[0], str):
                parts = value[0].split(".")
                cls = ROOT
                while parts:
                    part = parts.pop(0)
                    if not hasattr(cls, part):
                        cls = None
                        break
                    else:
                        cls = getattr(cls, part)
                if cls is not None:
                    setter(cls(*value[1:]))
                    continue

            setter(*value)
        else:
            setter(value)


def create_cms_labels(postfix="private work", x=0.135, y=0.96):
    cms = ROOT.TLatex(x, y, "HGCAL")
    setup_label(cms, {"TextFont": 73})
    label = ROOT.TLatex(x, y, "#font[73]{HGCAL} " + postfix)
    setup_label(label)
    return cms, label


def create_campaign_label(text, x=0.9625, y=0.96):
    label = ROOT.TLatex(x, y, text)
    setup_label(label, {"TextAlign": 31})
    return label