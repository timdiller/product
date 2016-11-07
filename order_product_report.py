# -*- coding: utf-8 -*-
from numpy import arange, array, logical_and, logical_or
from pandas import DataFrame, read_table

from chaco.api import ArrayPlotData, Plot
from enable.api import ComponentEditor
from traits.api import (
    Array, DelegatesTo, Enum, HasTraits, File, Instance, List, Property, Str,
    cached_property, on_trait_change,
)
from traitsui.api import (
    CheckListEditor, Group, HGroup, Item, ObjectColumn,
    TableEditor, View,
)

ANY = u'Any'
DATE = u'Date placed'
PROD_CLASS = u'Product class'
PRODUCT = u'Product purchased'
PROD_QUANT = u'Product quantity'
SUB_TOT = u'Subtotal (excl. tax)'
SALE_TOT = u'Total incl Tax'


def load_data(filename):
    from catalyst.pandas.convert import to_datetime, to_int
    from catalyst.pandas.headers import get_clean_names

    data_frame = read_table(
        filename,
        delimiter=',', encoding='utf-8', skiprows=0,
        na_values=['N/A', 'n/a'], comment=None, header=0,
        thousands=None, skipinitialspace=True,
    )

    # Ensure stripping and uniqueness of column names
    data_frame.columns = get_clean_names(data_frame.columns)

    # Type conversions
    for column in [u'Subtotal (excl. tax)', u'Payment Ref', u'Total incl Tax']:
        data_frame[column] = to_int(data_frame[column])
    for column in [u'Date placed', u'Subscription expiry date']:
        data_frame[column] = to_datetime(data_frame[column])
    data_frame.set_index(u"Date placed", inplace=True)
    return data_frame


class Sale(HasTraits):
    def __repr__(self):
        return "{}(date={}, amount=${})".format(self.__class__.__name__,
                                                self.get(DATE)[DATE],
                                                self.get(SALE_TOT)[SALE_TOT])


class ProductSalesReport(HasTraits):
    product_classes = Property(List, depends_on='sales_raw')
    data_file = File()
    sales_raw = Property(Instance(DataFrame), depends_on='data_file')

    @cached_property
    def _get_sales_raw(self):
        return load_data(self.data_file)

    @cached_property
    def _get_product_classes(self):
        return list(set(data.sales_raw[PROD_CLASS]))


class SalesReportModelView(HasTraits):
    model = Instance(ProductSalesReport)

    plot = Instance(Plot)

    agg_period = Enum("daily", "weekly", "monthly")
    data_file = DelegatesTo('model')
    metric = Enum(SALE_TOT, PROD_QUANT)
    num_sales = Property(Array, depends_on='sales')
    product_classes = Property(List, depends_on="model")
    product_class = Str()
    products = Property(List, depends_on='product_class')
    product = Str()
    revenue = Property(Array, depends_on='sales')
    sales = Property(Instance(DataFrame),
                     depends_on=['product_class', 'product'])
    sales_records = Property(List, depends_on=['product_class', 'product'])
    table_columns = Property(List, depends_on='model')

    @cached_property
    def _get_product_classes(self):
        return [ANY] + self.model.product_classes

    @cached_property
    def _get_products(self):
        if self.product_class == ANY:
            classes = self.model.sales_raw[PRODUCT]
        else:
            mask = self.model.sales_raw[PROD_CLASS] == self.product_class
            classes = self.model.sales_raw[PRODUCT][mask]
        return [ANY] + list(set(classes))

    @cached_property
    def _get_revenue(self):
        # TODO add date ranges here
        if SALE_TOT not in self.sales:
            return array([])
        else:
            rev = self.sales[SALE_TOT]
            revenue_array = rev.resample("W").sum().cumsum()
            return revenue_array.values

    @cached_property
    def _get_num_sales(self):
        # TODO add date ranges here
        if PROD_QUANT not in self.sales:
            return array([])
        else:
            rev = self.sales[PROD_QUANT]
            revenue_array = rev.resample("W").sum()
            return revenue_array.values

    @cached_property
    def _get_sales(self):
        mask = logical_and(
            logical_or(
                self.product_class == ANY,
                self.model.sales_raw[PROD_CLASS] == self.product_class
            ),
            logical_or(
                self.product == ANY,
                self.model.sales_raw[PRODUCT] == self.product
            )
        )
        return self.model.sales_raw[mask]

    @cached_property
    def _get_sales_records(self):
        columns = self.model.sales_raw.columns
        return [Sale(**{col: sale[col] for col in columns})
                for _, sale in self.sales.iterrows()]

    @cached_property
    def _get_table_columns(self):
        return [ObjectColumn(name=name) for name
                in self.model.sales_raw.columns.drop([PROD_CLASS])]

    def _plot_default(self):
        plot_data = ArrayPlotData(metric=self.revenue,
                                  date=arange(len(self.revenue)))
        plot = Plot(plot_data)
        plot.plot(("date", "metric"), type="bar", bar_width=0.8)
        return plot

    def default_traits_view(self):
        return View(
            Group(
                HGroup(
                    Item('data_file', show_label=False),
                    Group(
                        Item("product_class",
                             editor=CheckListEditor(name="product_classes")),
                        Item("product",
                             editor=CheckListEditor(name="products")),
                    ),
                ),
                Item("sales_records", editor=TableEditor(
                    columns=self.table_columns,
                    ),
                ),
                Item("metric", show_label=False),
                Item("plot", editor=ComponentEditor(), show_label=False),
            ),
            resizable=True,
        )

    @on_trait_change("product, product_class, metric")
    def update(self):
        if self.metric == SALE_TOT:
            data = self.revenue
        elif self.metric == PROD_QUANT:
            data = self.num_sales
        self.plot.data.set_data("metric", data)
        self.plot.data.set_data("date", arange(len(self.revenue)))


if __name__ == "__main__":
    from os.path import expanduser, join
    filename = expanduser(join(
        '~',
        'Downloads',
        'order-product-report-2015-11-01-to-2016-10-31.csv'
    ))
    data = ProductSalesReport(data_file=filename)
    print data.sales_raw.columns
    report = SalesReportModelView(model=data)
    report.configure_traits()
