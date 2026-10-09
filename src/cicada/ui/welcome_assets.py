"""蝉标志静态资产 (tui-startup-welcome 票据 02): 28×6 色格与 ASCII 轮廓.

选定的蓝白侧面蝉一次固化为受控静态数据 (设计版本 tui-startup-welcome-v1-20261008-r1,
来源文件与 hash 单列在设计 source-map, 运行时不回读)。生产运行只使用本模块常量:
字符白名单为空格/▀/▄/█, 颜色白名单为 blue/navy/white 或 None; 不读取设计目录、
原图或任何运行时图片转换, 不新增图片依赖。
"""

from __future__ import annotations

# 色名 -> #rrggbb (来源 palette: blue (35,103,255) / navy (21,53,147) / white (255,255,255))
LOGO_PALETTE: dict[str, str] = {
    "blue": "#2367ff",
    "navy": "#153593",
    "white": "#ffffff",
}

# 28 列 × 6 行色格: 每格 (字符, 前景色名, 背景色名); None = 无着色
LOGO_CELLS_28: tuple[tuple[tuple[str, str | None, str | None], ...], ...] = (
    (
        (' ', None, None), (' ', None, None), (' ', None, None), (' ', None, None),
        (' ', None, None), (' ', None, None), (' ', None, None), (' ', None, None),
        (' ', None, None), (' ', None, None), ('▄', 'blue', None), ('▄', 'blue', None),
        ('▄', 'blue', None), ('▄', 'blue', None), ('▄', 'blue', None), ('▄', 'blue', None),
        ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None),
        ('█', 'blue', None), ('▄', 'blue', None), ('▄', 'blue', None), (' ', None, None),
        (' ', None, None), ('▄', 'blue', None), (' ', None, None), ('▀', 'blue', None),
    ),
    (
        (' ', None, None), (' ', None, None), (' ', None, None), (' ', None, None),
        (' ', None, None), ('▄', 'blue', None), ('▄', 'blue', None), ('▀', 'blue', 'white'),
        ('▀', 'blue', 'white'), ('▀', 'blue', 'white'), ('█', 'white', None), ('█', 'white', None),
        ('█', 'white', None), ('█', 'white', None), ('█', 'white', None), ('▀', 'blue', 'white'),
        ('▀', 'blue', 'white'), ('▄', 'white', None), ('▀', 'blue', None), ('█', 'blue', None),
        ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None),
        ('█', 'blue', None), ('▀', 'blue', 'navy'), ('▀', 'navy', 'white'), ('▄', 'blue', None),
    ),
    (
        (' ', None, None), (' ', None, None), ('▄', 'blue', None), ('▀', 'blue', 'white'),
        ('▀', 'blue', 'white'), ('█', 'white', None), ('█', 'white', None), ('█', 'white', None),
        ('█', 'white', None), ('█', 'white', None), ('█', 'white', None), ('█', 'white', None),
        ('█', 'white', None), ('█', 'white', None), ('█', 'white', None), ('█', 'white', None),
        ('▀', 'white', 'blue'), ('▀', 'white', 'blue'), ('▄', 'blue', None), ('▀', 'blue', None),
        ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None),
        ('█', 'blue', None), ('█', 'navy', None), ('▀', 'blue', 'navy'), ('█', 'navy', None),
    ),
    (
        ('▄', 'blue', None), ('▀', 'blue', 'white'), ('█', 'white', None), ('█', 'white', None),
        ('█', 'white', None), ('█', 'white', None), ('█', 'white', None), ('█', 'white', None),
        ('█', 'white', None), ('█', 'white', None), ('█', 'white', None), ('▀', 'white', 'blue'),
        ('▀', 'white', 'blue'), ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None),
        ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None),
        ('█', 'blue', None), ('█', 'blue', None), ('▀', 'blue', None), ('▀', 'blue', None),
        ('█', 'blue', None), ('█', 'blue', None), ('▀', 'navy', None), (' ', None, None),
    ),
    (
        ('▀', 'blue', None), ('▀', 'white', 'blue'), ('▀', 'white', 'blue'), ('▀', 'white', 'blue'),
        ('▀', 'white', 'blue'), ('▀', 'white', 'blue'), ('▀', 'white', 'blue'), ('█', 'blue', None),
        ('▀', 'blue', None), ('▀', 'blue', None), ('█', 'blue', None), ('█', 'blue', None),
        ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None),
        ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None), ('█', 'blue', None),
        ('▀', 'blue', None), ('▀', 'blue', None), ('█', 'blue', None), ('▄', 'blue', None),
        (' ', None, None), (' ', None, None), (' ', None, None), (' ', None, None),
    ),
    (
        (' ', None, None), (' ', None, None), (' ', None, None), (' ', None, None),
        (' ', None, None), (' ', None, None), (' ', None, None), (' ', None, None),
        (' ', None, None), (' ', None, None), (' ', None, None), (' ', None, None),
        (' ', None, None), (' ', None, None), ('▀', 'blue', None), (' ', None, None),
        (' ', None, None), ('▄', 'blue', None), ('█', 'blue', None), (' ', None, None),
        (' ', None, None), (' ', None, None), (' ', None, None), (' ', None, None),
        ('█', 'blue', None), ('▄', 'blue', None), (' ', None, None), (' ', None, None),
    ),
)

# 无色模式 28×6 ASCII 轮廓 (行尾空格在渲染时补齐, 不在字面量保留)
LOGO_ASCII_28: tuple[str, ...] = (
    '                #######',
    '      ##..........#######@#',
    '  ##.............########@#@',
    '##..........##############@',
    '##.################# ##',
    '                  #     #',
)

__all__ = ["LOGO_ASCII_28", "LOGO_CELLS_28", "LOGO_PALETTE"]
