try: paraview.simple
except: from paraview.simple import *
paraview.simple._DisableFirstRenderCameraReset()

membrane000000_vtu = XMLUnstructuredGridReader( FileName=['/home/hpl/vc/fenics-tutorial/doc/src/src/membrane000000.vtu'] )

RenderView1 = GetRenderView()
RenderView1.CameraPosition = [0.0, 0.0, 10000.0]
RenderView1.CenterAxesVisibility = 0
RenderView1.InteractionMode = '2D'

membrane000000_vtu.PointArrayStatus = ['w']

DataRepresentation1 = Show()
DataRepresentation1.EdgeColor = [0.0, 0.0, 0.5000076295109483]
DataRepresentation1.SelectionPointFieldDataArrayName = 'w'
DataRepresentation1.ColorArrayName = ('POINT_DATA', 'w')
DataRepresentation1.ScalarOpacityUnitDistance = 0.2471957818677924
DataRepresentation1.ScaleFactor = 0.2

a1_w_PVLookupTable = GetLookupTableForArray( "w", 1, RGBPoints=[0.0, 0.23, 0.299, 0.754, 0.02894041074565366, 0.865, 0.865, 0.865, 0.05788082149130732, 0.706, 0.016, 0.15], VectorMode='Magnitude', NanColor=[0.25, 0.0, 0.0], ColorSpace='Diverging', ScalarRangeInitialized=1.0 )

a1_w_PiecewiseFunction = CreatePiecewiseFunction( Points=[0.0, 0.0, 0.5, 0.0, 0.05788082149130732, 1.0, 0.5, 0.0] )

DataRepresentation1.ScalarOpacityFunction = a1_w_PiecewiseFunction
DataRepresentation1.LookupTable = a1_w_PVLookupTable

a1_w_PVLookupTable.ScalarOpacityFunction = a1_w_PiecewiseFunction

RenderView1.CameraPosition = [0.0, 0.0, 5.464101615137755]
RenderView1.CameraClippingRange = [5.409460598986378, 5.546063139364821]
RenderView1.CameraParallelScale = 1.4142135623730951

Contour1 = Contour( PointMergeMethod="Uniform Binning" )

Contour1.PointMergeMethod = "Uniform Binning"
Contour1.ContourBy = ['POINTS', 'w']
Contour1.Isosurfaces = [0.02894041074565366]

Contour1.Isosurfaces = [0.01, 0.02, 0.03, 0.04]

DataRepresentation2 = Show()
DataRepresentation2.ColorArrayName = ('POINT_DATA', '')
DataRepresentation2.ScaleFactor = 0.11395682568800175
DataRepresentation2.SelectionPointFieldDataArrayName = 'w'
DataRepresentation2.EdgeColor = [0.0, 0.0, 0.5000076295109483]

DataRepresentation1.Visibility = 0

DataRepresentation1.Visibility = 1

ScalarBarWidgetRepresentation1 = CreateScalarBar( Title='w', LabelFontSize=12, Enabled=1, LookupTable=a1_w_PVLookupTable, TitleFontSize=12 )
GetRenderView().Representations.append(ScalarBarWidgetRepresentation1)

RenderView1.RemoteRenderThreshold = 3.0

a1_w_PVLookupTable.RGBPoints = [0.0, 1.0, 1.0, 0.4980392156862745, 0.02894041074565366, 1.0, 0.0, 0.4980392156862745, 0.05788082149130732, 0.6666666666666666, 0.3333333333333333, 0.0]

Render()
raw_input()
