# References

Add any external references here. This could be links to other documentation, or other resources that are relevant to the project.

## Websites
1. [How to deal with github commit failures due to oversized files](
https://blog.csdn.net/wxc1172300310/article/details/92799062?ops_request_misc=%257B%2522request%255Fid%2522%253A%2522a87c8851f49d13c05db1dc7a2d7b34f7%2522%252C%2522scm%2522%253A%252220140713.130102334.pc%255Fall.%2522%257D&request_id=a87c8851f49d13c05db1dc7a2d7b34f7&biz_id=0&utm_medium=distribute.pc_search_result.none-task-blog-2~all~first_rank_ecpm_v1~rank_v31_ecpm-3-92799062-null-null.142^v100^pc_search_result_base7&utm_term=github提交文件太大了&spm=1018.2226.3001.4187)
- Chapter 4 Soil Properties. (n.d.). Available at: https://hydrology.usu.edu/rrp/pdfs/ch4.pdf [Accessed 4 Sep. 2023].
- Geotech Data (n.d.). Soil permeability coefficient. [online] www.geotechdata.info. Available at: https://www.geotechdata.info/parameter/permeability.
- User, S. (n.d.). Soil porosity. [online] www.geotechdata.info. Available at: https://www.geotechdata.info/parameter/soil-porosity.

## Books
- Das, B.M. (2019). Advanced soil mechanics. Boca Raton: CRC Press, Taylor & Francis Group.
- Koestel, J., Larsbo, M. and Jarvis, N. (2020). Scale and REV analyses for porosity and pore connectivity measures in undisturbed soil. Geoderma, 366, p.114206. doi:https://doi.org/10.1016/j.geoderma.2020.114206.

## Journal Articles

## AI usage
1. [visualization](https://chatgpt.com/share/673f6e3b-2194-8006-87ab-9d960ce5d22a)
    - ask how to deal with postcode, for example: sector and district, standardize the format of postcode, sector and district

        - post code in the form of xxx xxx, how to divide it into two columns according to spaces
        - How to separate the last two digits of a string to form two columns
        - The fields in the file are of the form xxx x and xxxx x. Now I need to keep all the intervening spaces as one.
        - Replace consecutive spaces with cells

    - Interpretation of raw data, interpretation of field values in wet_day and typical_day
        - Are masd and mm and m and maod units converted?
        - What is the meaning of Downstream Stage and stage and tiding level in qualifier?
        - Stage Tipping Bucket Raingauge Downstream Stage Tidal Level 1 Water Height 2 Crest Tapping. What does each mean when the value in the qualifier is above?

    - Some basic questions:
        - Convert all of a column to a string
        - Convert all of a column to a value, if not successful, null value
        - How to show all the values in print
    
    - Data connection problems: abnormal columns after merge
        - Why is there a column without name in the last column after merge?
        - After two merges of data, there is a duplicate column with no column name, which is the connected column, how to deal with it?
        - Two merges were performed with different join columns, and on the second merge, the join column is duplicated in the dataset without a column name.
        - After grouping the data according to 'stationReference', 'parameter', 'unitName', grouped_data has columns 'stationReference', 'parameter', 'unitName', but after merging grouped_data with the station file merge, the resulting merge_data does not have the ' 'parameter', 'unitName' columns. Why and how to add it?

    - Error Handling
        - TypeError: 'ABCMeta' object is not subscriptable What does it mean?
        - MemoryError: Unable to allocate 781. PiB for array of shape (199667888, 550320386) and data type float64.

    - Drawing Maps:
        - How to stack multiple maps on top of each other to be able to show by legend selection
        - What are the common ones for cmap?
        - Are there other options for sequential types?
        - X, Y, np.where(Z > 0, Z, np.nan).T, cmap = plt.cm.Oranges, norm=matplotlib.colours.Normalize(vmin=Z.min(), vmax=Z.max()) What can be substituted for normalize in this?