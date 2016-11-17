# -*- coding: utf-8 -*-
import logging

from numpy import arange, ceil, floor, log10, logical_and, logical_or, zeros
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

AGG_PERIODS = {'Daily': 'D', 'Weekly': 'W', 'Monthly': 'M', 'Yearly': 'A'}
ANY = u'Any'
DATE = u'Date placed'
PROD_CLASS = u'Product class'
PRODUCT = u'Product purchased'
PROD_QUANT = u'Product quantity'
SALE_TOT = u'Total incl Tax'


def nearest_display_max(ary, div=2):
    """Return the next "Round" maximum of the given array
    """
    o_of_m = 10 ** floor(log10(max(ary)))
    mantissa = ceil(1 + div * max(ary) / o_of_m) / div
    return o_of_m * mantissa


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
    data_frame.set_index(u"Date placed", inplace=True, drop=False)
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

    agg_period = Enum(AGG_PERIODS.keys())
    data_file = DelegatesTo('model')
    metric = Enum(SALE_TOT, PROD_QUANT)
    num_sales = Property(Array, depends_on=['sales', 'agg_period',
                                            'model.data_file'])
    product_classes = Property(List, depends_on="model.product_classes")
    product_class = Str(ANY)
    products = Property(List, depends_on=['product_class', 'model.data_file'])
    product = Str(ANY)
    revenue = Property(Array, depends_on=['sales', 'agg_period',
                                          'model.data_file'])
    sales = Property(Instance(DataFrame),
                     depends_on=['product_class', 'product', 'model.data_file']
                     )
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
        logger.debug("Getting revenue")
        if SALE_TOT not in self.sales:
            logger.debug("{} not found in {}".format(SALE_TOT, self.sales))
            return zeros((10,))
        else:
            rev = self.sales[SALE_TOT]
            resampled_revenue = rev.resample(AGG_PERIODS[self.agg_period])
            return resampled_revenue.sum().fillna(0).cumsum().values

    @cached_property
    def _get_num_sales(self):
        # TODO add date ranges here
        if PROD_QUANT not in self.sales:
            logger.debug("{} not found in {}".format(PROD_QUANT, self.sales))
            return zeros((10,))
        else:
            num = self.sales[PROD_QUANT]
            resampled_num_sales = num.resample(AGG_PERIODS[self.agg_period])
            return resampled_num_sales.sum().fillna(0).values

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
        logger.debug("Found {} distinct sales records".format(mask.sum()))
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
        plot_data = ArrayPlotData(date=arange(len(self.revenue)))
        plot_data.set_data(SALE_TOT, self.revenue)
        plot_data.set_data(PROD_QUANT, self.num_sales)
        plot = Plot(plot_data)
        plot.plot(("date", SALE_TOT), type="line", name=SALE_TOT,
                  visible=self.metric == SALE_TOT)
        plot.plot(("date", PROD_QUANT), type="bar", bar_width=0.8,
                  name=PROD_QUANT, visible=self.metric == PROD_QUANT)
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
                    ), show_label=False,
                ),
                HGroup(
                    Item("metric", show_label=False),
                    Item("agg_period", show_label=False),
                ),
                Item("plot", editor=ComponentEditor(), show_label=False),
            ),
            resizable=True,
        )

    @on_trait_change("agg_period, product, product_class, metric, "
                     "model.data_file")
    def update(self):
        logger.debug("self.metric={!r}".format(self.metric))
        logger.debug("class={!r}".format(self.product_class))
        logger.debug("product={!r}".format(self.product))
        for name, renderer_list in self.plot.plots.items():
            renderer = renderer_list[0]
            visible = name == self.metric
            renderer.visible = visible
            logger.debug("Setting visible={} renderer {!r} ".format(visible,
                                                                    name))
        self.plot.data.set_data(SALE_TOT, self.revenue)
        self.plot.data.set_data(PROD_QUANT, self.num_sales)
        self.plot.data.set_data("date", arange(len(self.revenue)))
        disp_max = nearest_display_max(self.plot.data.arrays[self.metric])
        self.plot.range2d.set_bounds(('auto', 0.0), ('auto', disp_max))
        logger.debug("len(revenue)={}".format(len(self.revenue)))


if __name__ == "__main__":
    fmt = "%(asctime)s %(levelname)-8.8s [%(name)s:%(lineno)4s] %(message)s"
    logger = logging.getLogger(__name__)
    logger.setLevel(logging.DEBUG)
    if len(logger.handlers) < 2:
        formatter = logging.Formatter(fmt)
        file_handler = logging.FileHandler("prod.log")
        file_handler.setFormatter(formatter)
        stream_handler = logging.StreamHandler()
        stream_handler.setFormatter(formatter)
        logger.addHandler(file_handler)
        logger.addHandler(stream_handler)

    from os.path import expanduser, join
    filename = expanduser(join(
        '~',
        'Downloads',
        'order-product-report-2016-11-01-to-2016-11-08.csv',
        # 'order-product-report-2015-11-01-to-2016-10-31.csv'
    ))
    data = ProductSalesReport(data_file=filename)
    print data.sales_raw.columns
    report = SalesReportModelView(model=data)
    report.configure_traits()
