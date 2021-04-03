import pandas as pd
import plotly.express as px
import dash
import dash_core_components as dcc
import dash_html_components as html
from dash.dependencies import Input, Output
import numpy as np
import plotly.figure_factory as ff
import math

Playername=[]

fake=[[0.0]*13 for i in range(13)]
df = pd.read_csv('joint_salary.csv')
df2= pd.read_csv('averages.csv')
df3=pd.read_csv('yaxis.csv')
external_stylesheets = ['https://codepen.io/chriddyp/pen/bWLwgP.css']

app = dash.Dash(__name__, external_stylesheets=external_stylesheets)


def coef1(table1, table2):
    try:
        xsum = 0.0
        ysum = 0.0
        xysum = 0.0
        x2sum = 0.0
        y2sum = 0.0
        i = 0
        while i < len(table1):
            xsum = xsum + float(table1[i])
            ysum = ysum + float(table2[i])
            xysum = xysum + (float(table1[i]) * float(table2[i]))
            x2sum = x2sum + (float(table1[i]) * float(table1[i]))
            y2sum = y2sum + (float(table2[i]) * float(table2[i]))
            i = i + 1
        return (float(len(table1)) * xysum - xsum * ysum) / pow(
            ((float(len(table1)) * x2sum) - pow(xsum, 2)) * ((float(len(table2)) * y2sum) - pow(ysum, 2)), 0.5)
    except ZeroDivisionError:
        return 0


def coef(table1, table2):
    try:
        fake=[[0.0]*13 for i in range(13)]
        i = 0
        while i < len(table1):
            j = 0
            while j < len(table1):
                fake[i][j] = (coef1(table1[i], table2[j]))
                j = j + 1
            i = i + 1
        return fake
    except IndexError:
        fale= [[0.0] * 13 for i in range(13)]
        return fale

app.title='PERFORMANCE PROGRESS'
app.layout = html.Div([
    html.Div([
        #drop down menu and yaxis
        #dis plot and scatter plot
        html.Div([
            html.Div([
                html.H4(
                    'NBA Performance Progress Dashboard'),
                html.P(
                    'Filter by Teams:',
                    className="control_label"),
                  #drop down
                dcc.Dropdown(
                    id='my-dropdown',
                    options=[{'label': x, 'value': x} for x in sorted(df.Tm.unique())],
                    value=['GSW', 'HOU','LAL',
                           'MIA',
                           'MIL'],
                    multi=True,
                    className="dcc_control"),
                #yaxis
                html.P(
                    'Change Y-axis Variable: ',
                    className="control_label"
                ),
                dcc.Dropdown(
                    id='y-dropdown',
                    options=[{'label': x, 'value': x} for x in sorted(df3.yaxis)],
                    value='3P',
                    className="dcc_control")
            ],className='two columns'),

            html.Div([
                dcc.Graph(id='graph-output', figure={})
            ], className='five columns'),
            html.Div([
                dcc.Graph(id='g2', figure={})
            ], className='five columns')

        ],className='row'),
        #two matrix
        html.Div([
            #first matrix
            html.Div([
                dcc.Graph(id='heat_average', figure={})
            ], className='four columns'),
            #second matrix
            html.Div([
                dcc.Graph(id='heat_player', figure={})
            ], className='four columns'),
            html.Div([
                dcc.Graph(id='distributon_plot', figure={})
            ], className='four columns')
        ], className='row')
    ],id="mainContainer",
    style={
        "display": "flex",
        "flex-direction": "column"
    })
])


# Single Input, single Output, State, prevent initial trigger of callback, PreventUpdate

@app.callback(
    Output(component_id='graph-output', component_property='figure'),
    Input(component_id='my-dropdown', component_property='value'),
    Input(component_id='y-dropdown', component_property='value'),
    prevent_initial_call=False
)
def update_my_graph(val_chosen,y_axis_val):
    if len(val_chosen) > 0:
        # print(n)
        df['Year'] = df['Year'].astype(str)
        dff = df[df["Tm"].isin(val_chosen)]

        figure = px.scatter(dff, x="PTS", y=str(y_axis_val), color='Year', hover_name='Player', size='Salary',
                            labels={'PTS': 'Total Points',
                                    str(y_axis_val): str(y_axis_val),
                                    'Tm': 'Team'
                                    },
                            title=str(y_axis_val)+' Compared to Total Points Made by Years',

                            )

        figure.update_layout(dragmode='select',title_x=0.5, font_family="Garamond")
        return figure

    elif len(val_chosen) == 0:
        raise dash.exceptions.PreventUpdate

@app.callback(
    Output(component_id='distributon_plot', component_property='figure'),
    dash.dependencies.Input(component_id='graph-output', component_property='hoverData'),
    prevent_initial_call=False
)
def displot(hoverData):
    global df

    if not hoverData:
        Player=''
    else:
        Player = hoverData['points'][0]['hovertext'].encode('utf-8')

    filterDF = df[df.Player == Player]
    if Player=='':
        name="Player Turnovers to Assists"
    else:
        name=str(Player)+" Turnovers to Assists"
    figure = px.density_contour(filterDF, x="AST", y="TOV", title=name)
    figure.update_traces(contours_coloring="fill", contours_showlabels=True)
    figure.update_layout(title_x=0.5, font_family="Garamond")
    return figure

def managerPlayer(Player):
    global Playername
    if len(Playername)<2:
        Playername.append(Player)
    else:
        Playername.pop(0)
        Playername.append(Player)

@app.callback(
    Output(component_id='heat_average', component_property='figure'),
    dash.dependencies.Input(component_id='graph-output', component_property='hoverData'),
    prevent_initial_call=False
)

def heat_average(hoverData):
    global df
    global df2
    global fake
    if not hoverData:
        Player = ''
    else:
        Player = hoverData['points'][0]['hovertext'].encode('utf-8')
    if Player=='' :
        e=fake
    else:
        try:
            a = df.loc[df['Player'] == Player][
                ['GS', 'MP', '3P%', '2P%', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'PF', 'PTS', 'Salary']].values
            b= df2[['GS','MP','3P%','2P%','FT%','ORB','DRB','AST','STL','BLK','PF','PTS','Salary']].values
            c = np.array(a).T.tolist()
            d = np.array(b).T.tolist()
            e=coef(c,d)[::-1]
        except ValueError or IndexError:
            e=fake
    if Player=='':
        name1='Player'
        name= 'Player-to-Average'
    else:
        name1=str(Player)
        name= '['+name1+']-to-Average'
    fig= px.imshow(e,color_continuous_scale='RdBu_r', title=name,
                   labels=dict(y=name1,x="Average Player"),
                   x=['GS', 'MP', '3P%', '2P%', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'PF', 'PTS', 'Salary'],
                   y=['GS', 'MP', '3P%', '2P%', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'PF', 'PTS', 'Salary'][::-1],
                   )
    fig.update_layout(title_x=0.5, font_family="Garamond")
    return fig

@app.callback(
    Output(component_id='heat_player', component_property='figure'),
    dash.dependencies.Input(component_id='graph-output', component_property='hoverData'),
    prevent_initial_call=False
)

def heat_player(hoverData):
    global df
    global Playername
    global fake
    if not hoverData:
        Player = ''
    else:
        Player = hoverData['points'][0]['hovertext'].encode('utf-8')
        managerPlayer(Player)
    if Player=='' or len(Playername)<2:
        e=fake
    else:
        try:
            a = df.loc[df['Player'] == Playername[0]][
                ['GS', 'MP', '3P%', '2P%', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'PF', 'PTS', 'Salary']].values
            b = df.loc[df['Player'] == Playername[1]][
                ['GS', 'MP', '3P%', '2P%', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'PF', 'PTS', 'Salary']].values
            c = np.array(a).T.tolist()
            d = np.array(b).T.tolist()
            e=coef(c,d)[::-1]
        except ValueError or IndexError:
            e=fake

    if len(Playername)==2:
        name1=str(Playername[0])
        name2=str(Playername[1])
        name= "["+name1+"]-to-["+name2+"]"
    else:
        name1="Player"
        name2='Player'
        name= 'Player-to-Player'
    fig= px.imshow(e,color_continuous_scale='RdBu_r', title=name,
                   labels=dict(x=name2,y=name1),
                   x=['GS', 'MP', '3P%', '2P%', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'PF', 'PTS', 'Salary'],
                   y=['GS', 'MP', '3P%', '2P%', 'FT%', 'ORB', 'DRB', 'AST', 'STL', 'BLK', 'PF', 'PTS', 'Salary'][::-1],
                   )
    #labels={'GS':'GS', 'MP':'MP', '3P%':'3P%', '2P%':'2P%', 'FT%':'FT%', 'ORB':'ORB', 'DRB':'DRB', 'AST':'AST', 'STL':'STL', 'BLK':'BLK', 'PF':'PF', 'PTS':'PTS', 'Salary':'Salary'}
    fig.update_layout(title_x=0.5,font_family="Garamond")
    return fig

@app.callback(
    Output('g2', 'figure'),
    Input(component_id='y-dropdown', component_property='value'),
    dash.dependencies.Input('graph-output', 'selectedData'),
    Input('g2', 'selectedData'),
    prevent_initial_call=False
)
def copychart(value, selectedData, data):
    global df
    Player=[]
    lines = {}
    if selectedData:
        for selected_data in [selectedData]:
            for selected_points in selectedData['points']:
                lines[selected_points['hovertext'].encode('utf-8')]= selected_points['y']
                Player.append(selected_points['hovertext'].encode('utf-8'))
        df1 = df[df["Player"].isin(Player)]
        x1=[]
        x1.append(df[str(value)].tolist())
        fig = ff.create_distplot(x1,[str(value)], show_rug=False, show_hist=False)
        # add lines using absolute references
        if data:
            for k in lines.keys():
                # print(k)
                fig.add_shape(type='line',
                              yref="y",
                              xref="x",
                              x0=lines[k],
                              y0=data['range']['y'][0],
                              x1=lines[k],
                              y1=data['range']['y'][1],
                              line=dict(color='black', width=1))
                fig.add_annotation(
                    x=lines[k],
                    y=1.06,
                    yref='paper',
                    showarrow=False,
                    text=k)
        fig.update_layout(dragmode='select', title= 'KDE Curve',title_x=0.5, font_family="Garamond")
        return fig
    else:
        raise dash.exceptions.PreventUpdate


if __name__ == '__main__':
    app.run_server(debug=True)
