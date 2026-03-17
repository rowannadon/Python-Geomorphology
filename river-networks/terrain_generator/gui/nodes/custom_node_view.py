"""Custom node view with dynamic border width support."""

from Qt import QtCore, QtGui
from NodeGraphQt.qgraphics.node_base import NodeItem
from NodeGraphQt.constants import NodeEnum


class CustomNodeItem(NodeItem):
    """Custom node item with dynamic border width."""
    
    def __init__(self):
        super().__init__()
        self._custom_border_width = 0.8  # Default width
    
    def set_border_width(self, width):
        """Set custom border width."""
        self._custom_border_width = width
        self.update()
    
    def paint(self, painter, option, widget):
        """Override paint to use custom border width."""
        self.auto_switch_mode()
        
        # Determine which paint method to use based on layout
        if self.layout_direction == 0:
            self._paint_horizontal_custom(painter, option, widget)
        else:
            self._paint_vertical_custom(painter, option, widget)
    
    def _paint_horizontal_custom(self, painter, option, widget):
        """Custom horizontal paint with dynamic border."""
        painter.save()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.NoBrush)

        # base background.
        margin = 1.0
        rect = self.boundingRect()
        rect = QtCore.QRectF(rect.left() + margin,
                             rect.top() + margin,
                             rect.width() - (margin * 2),
                             rect.height() - (margin * 2))

        radius = 4.0
        painter.setBrush(QtGui.QColor(*self.color))
        painter.drawRoundedRect(rect, radius, radius)

        # light overlay on background when selected.
        if self.selected:
            painter.setBrush(QtGui.QColor(*NodeEnum.SELECTED_COLOR.value))
            painter.drawRoundedRect(rect, radius, radius)

        # node name background.
        padding = 3.0, 2.0
        text_rect = self._text_item.boundingRect()
        text_rect = QtCore.QRectF(text_rect.x() + padding[0],
                                  rect.y() + padding[1],
                                  rect.width() - padding[0] - margin,
                                  text_rect.height() - (padding[1] * 2))
        if self.selected:
            painter.setBrush(QtGui.QColor(*NodeEnum.SELECTED_COLOR.value))
        else:
            painter.setBrush(QtGui.QColor(0, 0, 0, 80))
        painter.drawRoundedRect(text_rect, 3.0, 3.0)

        # node border - USE CUSTOM WIDTH
        if self.selected:
            border_width = max(1.2, self._custom_border_width)
            border_color = QtGui.QColor(*NodeEnum.SELECTED_BORDER_COLOR.value)
        else:
            border_width = self._custom_border_width
            border_color = QtGui.QColor(*self.border_color)

        border_rect = QtCore.QRectF(rect.left(), rect.top(),
                                    rect.width(), rect.height())

        pen = QtGui.QPen(border_color, border_width)
        pen.setCosmetic(self.viewer().get_zoom() < 0.0)
        path = QtGui.QPainterPath()
        path.addRoundedRect(border_rect, radius, radius)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(pen)
        painter.drawPath(path)

        painter.restore()
    
    def _paint_vertical_custom(self, painter, option, widget):
        """Custom vertical paint with dynamic border."""
        painter.save()
        painter.setPen(QtCore.Qt.NoPen)
        painter.setBrush(QtCore.Qt.NoBrush)

        # base background.
        margin = 1.0
        rect = self.boundingRect()
        rect = QtCore.QRectF(rect.left() + margin,
                             rect.top() + margin,
                             rect.width() - (margin * 2),
                             rect.height() - (margin * 2))

        radius = 4.0
        painter.setBrush(QtGui.QColor(*self.color))
        painter.drawRoundedRect(rect, radius, radius)

        # light overlay on background when selected.
        if self.selected:
            painter.setBrush(QtGui.QColor(*NodeEnum.SELECTED_COLOR.value))
            painter.drawRoundedRect(rect, radius, radius)

        # top & bottom edge background.
        padding = 2.0, 2.0
        height = self._text_item.boundingRect().height() + (padding[1] * 2)
        
        # top rect
        top_rect = QtCore.QRectF(rect.x(), rect.y(), rect.width(), height)
        painter.setBrush(QtGui.QColor(0, 0, 0, 80))
        painter.drawRoundedRect(top_rect, radius, radius)
        # draw bottom corners
        for pos in [top_rect.left(), top_rect.right() - 5.0]:
            painter.drawRect(QtCore.QRectF(pos, top_rect.bottom() - 5.0, 5.0, 5.0))
        
        # bottom rect
        bottom_rect = QtCore.QRectF(rect.x(), rect.bottom() - height, 
                                    rect.width(), height)
        painter.setBrush(QtGui.QColor(0, 0, 0, 80))
        painter.drawRoundedRect(bottom_rect, radius, radius)
        # draw top corners
        for pos in [bottom_rect.left(), bottom_rect.right() - 5.0]:
            painter.drawRect(QtCore.QRectF(pos, bottom_rect.top(), 5.0, 5.0))

        # node border - USE CUSTOM WIDTH
        if self.selected:
            border_width = max(1.2, self._custom_border_width)
            border_color = QtGui.QColor(*NodeEnum.SELECTED_BORDER_COLOR.value)
        else:
            border_width = self._custom_border_width
            border_color = QtGui.QColor(*self.border_color)

        border_rect = QtCore.QRectF(rect.left(), rect.top(),
                                    rect.width(), rect.height())
        pen = QtGui.QPen(border_color, border_width)
        pen.setCosmetic(self.viewer().get_zoom() < 0.0)
        path = QtGui.QPainterPath()
        path.addRoundedRect(border_rect, radius, radius)
        painter.setBrush(QtCore.Qt.NoBrush)
        painter.setPen(pen)
        painter.drawPath(path)

        painter.restore()